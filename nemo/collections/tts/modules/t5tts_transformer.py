#############################################################################
# Copyright (c) 2023 NVIDIA CORPORATION. All rights reserved.
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#############################################################################
import math

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_kvpacked_func, flash_attn_varlen_qkvpacked_func
from flash_attn.layers.rotary import RotaryEmbedding
from torch import nn
from torch.cuda import amp
from torch.nn.utils.rnn import pad_sequence
from nemo.utils import logging

# alibi helper functions (start)
def get_relative_positions(seq_len: int, device, symmetric: bool=True) -> torch.tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    rel_pos = (x - y).to(device)
    if symmetric:
        rel_pos = -rel_pos.abs()
    return rel_pos


def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
# alibi helper functions (end)

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='gpt2',
                 is_causal=False, norm_name=''):
        super(ConvNorm, self).__init__()

        padding = 0 if is_causal else padding
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.is_causal = is_causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias)

        if w_init_gain == 'gpt2':
            torch.nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(
                self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

        self.norm_name = norm_name
        if self.norm_name == 'weightnorm':
            self.conv = nn.utils.weight_norm(self.conv)
        elif self.norm_name == 'spectralnorm':
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, signal):
        if self.is_causal:
            padding = (int((self.kernel_size - 1) * (self.dilation)), 0)
            signal = torch.nn.functional.pad(signal, padding)
        conv_signal = self.conv(signal)
        return conv_signal
    
class LayerNorm(nn.Module):
    """ LayerNorm with optional bias. PyTorch doesn't support bias=False """
    def __init__(self, size, gamma0=1, eps=1e-5, use_bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size)) if use_bias else None
        self.eps = eps
        self.size = size

    def forward(self, tensor):
        """
        tensor (B, T, C)
        """
        dtype = tensor.dtype
        # fp32 to avoid numerical issues
        with amp.autocast(enabled=True, dtype=torch.float32):
            tensor = F.layer_norm(
                tensor, self.weight.shape, self.weight, self.bias, self.eps)
        return tensor.to(dtype)


class LayerScale(nn.Module):
    """
    # borrowed from https://huggingface.co/
    Layer scale from [Touvron et al 2021] (https://arxiv.org/abs/2103.17239).
    Rescales diagonaly the residual outputs close to 0, with a learnt scale.
    Args:
        channels (int): Number of channels.
        init (float): Initial scale.
        channel_last (bool): True expects `[*, C]` , otherwise, `[*, C, T]`.
        device (torch.device or None): Device to initialize the module.
        dtype (torch.dtype or None): dtype to use to initialize the module.
    """
    def __init__(self, channels: int, init: float = 1e-4, channel_last: bool =
                 True, device=None, dtype=None):
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(
            torch.full((channels,), init,
                       requires_grad=True, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        with amp.autocast(enabled=True, dtype=torch.float32):
            if self.channel_last:
                x = self.scale * x
            else:
                x = self.scale[:, None] * x
        return x.to(dtype)


class PositionwiseConvFF(nn.Module):
    def __init__(self, d_model, p_dropout, kernel_size=1, bias=False,
                 is_causal=True):
        super(PositionwiseConvFF, self).__init__()

        self.d_model = d_model
        self.non_linearity = nn.GELU(approximate="tanh")
        self.proj = ConvNorm(d_model, d_model * 4, bias=bias,
                             kernel_size=kernel_size, is_causal=is_causal)
        self.o_net = ConvNorm(d_model * 4, d_model, bias=bias,
                              kernel_size=kernel_size, is_causal=is_causal)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """
        x (B, T, C)
        """
        x = self.non_linearity(self.proj(x.transpose(1, 2)))
        x = self.dropout(self.o_net(x).transpose(1, 2))
        return x


class Attention(nn.Module):
    def __init__(self, n_heads, d_model, p_dropout, is_causal=True,
                 is_self_attention=True, d_memory=None,
                 use_flash_attention=True, deterministic=False,
                 pos_emb={"name": "alibi"}, max_length_causal_mask=4096):
        super(Attention, self).__init__()
        # context conditional attention dims
        if is_self_attention:
            assert d_model % n_heads == 0, "d_model % n_head != 0"
            self.d_head = d_model // n_heads
        else:
            assert d_memory % n_heads == 0, "d_memory % n_head != 0"
            self.d_head = d_memory // n_heads

        self.n_heads = n_heads
        self.d_model = d_model
        self.scale = self.d_head ** -0.5
        self.is_causal = is_causal
        self.is_self_attention = is_self_attention
        self.use_flash_attention = use_flash_attention
        self.deterministic = deterministic
        self.pos_emb_name = pos_emb['name']
        self.max_length_causal_mask = max_length_causal_mask
        if self.pos_emb_name == 'rope':
            self.rope = RotaryEmbedding(
                self.d_head, base=pos_emb['base'])
        elif self.pos_emb_name == 'alibi':
            self.register_buffer("m", get_alibi_slope(self.n_heads))
        elif self.pos_emb_name == 'learnable':
            # deprecated - use learnable_v2, pos emb should not be in each attention layer
            self.position_embeddings = nn.Embedding(max_length_causal_mask, d_model)

        if is_causal and is_self_attention:
            # ~ 45 seconds mask, 4096 mel frames, 86 frames per second
            # ~ 762 seconds mask, 65536 mel frames, 86 frames per second

            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(4096, 4096)).view(1, 1, 4096, 4096) == 0)

        if is_self_attention:
            self.qkv_net = nn.Linear(
                d_model, 3 * n_heads * self.d_head, bias=False)
            self.o_net = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        else:
            self.q_net = nn.Linear(d_model, n_heads * self.d_head, bias=False)
            self.kv_net = nn.Linear(d_memory, 2 * n_heads * self.d_head, bias=False)
            self.o_net = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        self.dropout = nn.Dropout(p_dropout)
        self.use_cache = False
    
    def reset_cache(self, use_cache=False):
        self.use_cache = use_cache
        self.cache = {
            'is_initialized': False,
            'self_k': None,
            'self_v': None,
            'cross_kv' : None,
            'cross_k': None,
            'cross_v': None
        }

    def add_positional_embeddings(self, x, start_step=0):
        # Deprecated - position embeddings are handled at the stack level
        positions = torch.arange(start_step, start_step+x.size(1), device=x.device).unsqueeze(0)
        pos_emb = self.position_embeddings(positions)
        return x + pos_emb
    
    def attn_flash(self, query, query_mask, memory=None, memory_mask=None,
                   idx=None):
        alibi_slopes = None
        if self.pos_emb_name == 'learnable':
            # deprecated - use learnable_v2, pos emb should not be in each attention layer
            query = self.add_positional_embeddings(query)
            if memory is not None:
                memory = self.add_positional_embeddings(memory)

        if self.is_self_attention:
            B, T, D = query.shape
            d_head = D // self.n_heads
            qkv = self.qkv_net(query).reshape(B, T, 3, self.n_heads, d_head)
            if self.pos_emb_name == 'rope':
                qkv = self.rope(qkv)
            elif self.pos_emb_name == 'alibi':
                alibi_slopes = self.m[:, 0, 0]
            qkv = qkv[~query_mask].reshape(-1, 3, self.n_heads, d_head)
            lengths_q = (~query_mask).sum(1)
            cu_seqlens_q = F.pad(lengths_q.cumsum(0), (1, 0), value=0).to(torch.int32)
            max_seqlen_q = torch.max(lengths_q)
            y = flash_attn_varlen_qkvpacked_func(
                qkv.bfloat16(), cu_seqlens=cu_seqlens_q,
                max_seqlen=max_seqlen_q, dropout_p=self.dropout.p,
                causal=self.is_causal, alibi_slopes=alibi_slopes,
                deterministic=self.deterministic)
        else:
            Bq, Tq, _ = query.shape
            Bkv, Tkv, _ = memory.shape
            q = self.q_net(query).reshape(Bq, Tq, self.n_heads, self.d_head)
            kv = self.kv_net(memory).reshape(Bkv, Tkv, 2, self.n_heads, self.d_head)
            if self.pos_emb_name == 'rope':
                q, kv = self.rope(q, kv)
            elif self.pos_emb_name == 'alibi':
                alibi_slopes = self.m[:, 0, 0]
            q = q[~query_mask].reshape(-1, self.n_heads, self.d_head)
            kv = kv[~memory_mask].reshape(-1, 2, self.n_heads, self.d_head)
            lengths_q = (~query_mask).sum(1)
            lengths_k = (~memory_mask).sum(1)
            cu_seqlens_q = F.pad(lengths_q.cumsum(0), (1, 0), value=0).to(torch.int32)
            cu_seqlens_k = F.pad(lengths_k.cumsum(0), (1, 0), value=0).to(torch.int32)
            max_seqlen_q = torch.max(lengths_q)
            max_seqlen_k = torch.max(lengths_k)
            y = flash_attn_varlen_kvpacked_func(
                q.bfloat16(), kv.bfloat16(), cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, dropout_p=self.dropout.p,
                causal=self.is_causal, alibi_slopes=alibi_slopes,
                deterministic=self.deterministic)
        # (rvalle): modify such that transformer uses no padding at all
        B, T, C = query.shape
        y = pad_sequence(torch.split(y, lengths_q.tolist()), batch_first=True)
        y = y.to(query.dtype).view(B, T, -1)
        return y

    def attn_naive(self, query, query_mask, memory=None, memory_mask=None,
                   attn_prior=None, dump_attention=False, idx=0):
        
        pos_start_time_step = 0
        if self.use_cache:
            if self.cache['is_initialized']:
                pos_start_time_step = query.size(1) - 1
                query = query[:, -1:, :]
                query_mask = query_mask[:, -1:]
            else:
                self.cache['is_initialized'] = True

        B, T, _ = query.shape
        Tkv = T if memory is None else memory.shape[1]
        mask = None
        
        if self.pos_emb_name == 'learnable':
            # deprecated - use learnable_v2, pos emb should not be in each attention layer
            query = self.add_positional_embeddings(query, pos_start_time_step)
            if memory is not None:
                memory = self.add_positional_embeddings(memory)

        if self.is_self_attention:
            qkv = self.qkv_net(query).reshape(B, T, 3, self.n_heads, self.d_head)
            qkv = self.rope(qkv) if self.pos_emb_name == 'rope' else qkv
            q, k, v = qkv.chunk(3, dim=2)
            q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
            if self.use_cache:
                if self.cache['self_k'] is not None:
                    k = torch.cat([self.cache['self_k'], k], dim=1)
                    v = torch.cat([self.cache['self_v'], v], dim=1)
                self.cache['self_k'] = k
                self.cache['self_v'] = v
        else:
            Bq, Tq, _ = query.shape
            Bkv, Tkv, _ = memory.shape
            q = self.q_net(query).reshape(Bq, Tq, self.n_heads, self.d_head)
            if self.use_cache and self.cache['cross_kv'] is not None:
                kv = self.cache['cross_kv']
            else:
                kv = self.kv_net(memory).reshape(Bkv, Tkv, 2, self.n_heads, self.d_head)
            q, kv = self.rope(q, kv) if self.pos_emb_name == 'rope' else q, kv
            if self.use_cache and self.cache['cross_k'] is not None:
                k = self.cache['cross_k']
                v = self.cache['cross_v']
            else:
                k, v = kv.chunk(2, dim=2)
                k, v = k.squeeze(2), v.squeeze(2)
                if self.use_cache:
                    self.cache['cross_kv'] = kv
                    self.cache['cross_k'] = k
                    self.cache['cross_v'] = v

        # (B, T, nh * dh) -> (B, nh, T, dh)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        bias = 0
        if self.pos_emb_name == 'alibi':
            rel_pos = get_relative_positions(max(T, Tkv), q.device)[:T, :Tkv]
            bias = (self.m * rel_pos).unsqueeze(0)

        attn_score = bias + torch.matmul(q, k.transpose(2, 3)) * self.scale

        if not self.is_self_attention and memory_mask is not None:
            mask = memory_mask[:, None, None]

        if self.is_self_attention and query_mask is not None:
            mask = query_mask[:, None, :, None]

        if mask is not None:
            # assumes there's at least one mask
            attn_score.masked_fill_(mask, float('-inf'))

        if self.is_self_attention and self.is_causal:
            attn_score.masked_fill_(
                self.causal_mask[..., :T, :T], float('-inf'))

        # attn_prior or square mask or vanilla attention
        if attn_prior is not None:
            eps = 1e-8
            attn_prior = attn_prior[:, :T]  # trim for inference
            attn_prior = torch.log(attn_prior + eps)
            attn_prior = attn_prior[:, None].repeat(1, self.n_heads, 1, 1)
            attn_score_log = F.log_softmax(attn_score, dim=-1) + attn_prior
            attn_prob = F.softmax(attn_score_log, dim=-1)
        else:
            attn_score_log = F.log_softmax(attn_score, dim=-1)
            _attn_score = attn_score_log
            attn_prob = F.softmax(attn_score, dim=-1)

        # replace inf and nans with 0.0
        if mask is not None:
            _attn_score = attn_score
            attn_prob = attn_prob.masked_fill(mask, 0.0)

        attn_prob = self.dropout(attn_prob)

        y = torch.matmul(attn_prob, v)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        return y, [attn_prob, _attn_score]

    def forward(self, query, query_mask=None, memory=None, memory_mask=None,
                attn_prior=None, dump_attention=False, idx=None):
        """
        all inputs should be (B, T, C)
        query_mask (T1, T1)
        memory_mask (B, T2)
        attn_prior (T1, T2)
        """

        if self.use_flash_attention:
            attn_prob = []
            y = self.attn_flash(query, query_mask, memory, memory_mask, idx)
        else:
            y, attn_prob = self.attn_naive(
                query, query_mask, memory, memory_mask, attn_prior,
                dump_attention, idx=idx)

        y = self.dropout(self.o_net(y))

        return y, attn_prob


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, kernel_size, p_dropout, context_xattn,
                 has_xattn, remove_self_attention=False, is_causal=True,
                 apply_norm_to_cond=True, layer_norm_method='pre',
                 use_layer_scale=False, layer_scale_init=1e-1,
                 use_flash_self_attention=True, use_flash_x_attention=True, deterministic=False,
                 pos_emb={"name": "alibi"}, max_length_causal_mask=4096):
        super(TransformerBlock, self).__init__()
        """
        T5-ish
        """
        self.layer_norm_method = layer_norm_method
        self.has_xattn = has_xattn
        self.remove_self_attention = remove_self_attention
        self.use_layer_scale = use_layer_scale

        if not self.remove_self_attention:
            self.norm_self = LayerNorm(d_model, use_bias=False)
            self.self_attention = Attention(
                n_heads=n_heads, d_model=d_model, p_dropout=p_dropout,
                is_self_attention=True,
                use_flash_attention=use_flash_self_attention,
                deterministic=deterministic, pos_emb=pos_emb,
                max_length_causal_mask=max_length_causal_mask)
            if self.use_layer_scale:
                self.layer_scale_self_attn = LayerScale(
                    d_model, init=layer_scale_init)
            else:
                self.layer_scale_self_attn = nn.Identity()

        if self.has_xattn:
            self.apply_norm_to_cond = apply_norm_to_cond
            self.norm_xattn_query = LayerNorm(d_model, use_bias=False)
            params = context_xattn['params']
            cross_attention = Attention(
                n_heads=params['n_heads'], d_model=d_model,
                p_dropout=p_dropout, is_causal=False,
                is_self_attention=False, d_memory=params['d_heads'],
                use_flash_attention=use_flash_x_attention,
                deterministic=deterministic,
                pos_emb=params.get('pos_emb', pos_emb),
                max_length_causal_mask=params.get(
                    'max_length_causal_mask', max_length_causal_mask)
            )
            if self.use_layer_scale:
                layer_scale_cross_attn = LayerScale(
                    d_model, init=layer_scale_init)
            else:
                layer_scale_cross_attn = nn.Identity()
            if self.apply_norm_to_cond:
                norm_xattn_memory = LayerNorm(
                    params['d_heads'], use_bias=False)

            if self.apply_norm_to_cond:
                self.norm_xattn_memory = norm_xattn_memory

            self.cross_attention = cross_attention
            self.layer_scale_cross_attn = layer_scale_cross_attn

        self.norm_pos_ff = LayerNorm(d_model, use_bias=False)
        self.pos_ff = PositionwiseConvFF(
            d_model, p_dropout, kernel_size=kernel_size, is_causal=is_causal)

        if self.use_layer_scale:
            self.layer_scale_ff = LayerScale(d_model, init=layer_scale_init)
        else:
            self.layer_scale_ff = nn.Identity()
        
        self.use_cache = False

    def reset_cache(self, use_cache=False):
        self.use_cache = use_cache
        self.cache = {
            'self_attn_output': None,
            'cross_attn_output': None,
            'memory': None,
        }
        self.self_attention.reset_cache(use_cache)
        if self.has_xattn:
            self.cross_attention.reset_cache(use_cache)

    def forward(self, x, x_mask, cond, cond_mask, dump_attention=False,
                attn_prior=None, idx=None):
        """
        all inputs should be (B, T, C)
        mask (T1, T2) is True where masking (ignoring) is required
        """
        x_mask_inv_float = (~x_mask).to(x.dtype)[..., None]
        if self.layer_norm_method == 'pre':
            x_, s_attn_prob = self.self_attention(
                query=self.norm_self(x),
                query_mask=x_mask,
                dump_attention=dump_attention, idx=idx)
            if self.use_cache:
                if self.cache['self_attn_output'] is not None:
                    x_ = torch.cat([self.cache['self_attn_output'], x_], dim=1)
                self.cache['self_attn_output'] = x_

            x = (x + self.layer_scale_self_attn(x_)) * x_mask_inv_float
        elif self.layer_norm_method == 'post':
            x_, s_attn_prob = self.self_attention(
                query=x,
                query_mask=x_mask,
                dump_attention=dump_attention)
            if self.use_cache:
                if self.cache['self_attn_output'] is not None:
                    x_ = torch.cat([self.cache['self_attn_output'], x_], dim=1)
                self.cache['self_attn_output'] = x_
            x = x + self.layer_scale_self_attn(x_)
            x = self.norm_self(x) * x_mask_inv_float

        x_attn_prob = None
        if self.has_xattn and cond is not None:
            if self.layer_norm_method == 'pre':
                x_normed = self.norm_xattn_query(x)
                if self.use_cache and self.cache['memory'] is not None:
                    memory = self.cache['memory']
                else:
                    memory = (self.norm_xattn_memory(cond)
                            if self.apply_norm_to_cond else cond)
                    if self.use_cache:
                        self.cache['memory'] = memory

                x_res, x_attn_prob = self.cross_attention(
                    query=x_normed,
                    query_mask=x_mask,
                    memory=memory,
                    memory_mask=cond_mask,
                    attn_prior=attn_prior,
                    dump_attention=dump_attention,
                    idx=idx)
                if self.use_cache:
                    if self.cache['cross_attn_output'] is not None:
                        x_res = torch.cat([self.cache['cross_attn_output'], x_res], dim=1)
                    self.cache['cross_attn_output'] = x_res
                x = x + self.layer_scale_cross_attn(x_res)  # unbounded
                x = x * x_mask_inv_float
            elif self.layer_norm_method == 'post':
                x_res, x_attn_prob = self.cross_attention(
                    query=x,
                    query_mask=x_mask,
                    memory=cond,
                    memory_mask=cond_mask,
                    attn_prior=attn_prior,
                    dump_attention=dump_attention)
                if self.use_cache:
                    if self.cache['cross_attn_output'] is not None:
                        x_res = torch.cat([self.cache['cross_attn_output'], x_res], dim=1)
                    self.cache['cross_attn_output'] = x_res
                    
                x = (x + self.layer_scale_cross_attn(x_res)) * x_mask_inv_float
                x = self.norm_xattn_query(x)

        # mlp final projection
        if self.layer_norm_method == 'pre':
            x = x + self.layer_scale_ff(
                self.pos_ff(self.norm_pos_ff(x)))
            x *= x_mask_inv_float
        elif self.layer_norm_method == 'post':
            x = x + self.layer_scale_ff(self.pos_ff(x))
            x *= x_mask_inv_float
            x = self.norm_pos_ff(x)

        attn_probabilities = {
            'self_attn_probabilities': s_attn_prob,
            'cross_attn_probabilities': x_attn_prob
        }

        return {
            'output': x,
            'attn_probabilities': attn_probabilities,
        }


class TransformerStack(nn.Module):
    def __init__(self,  hparams):
        super(TransformerStack, self).__init__()
        """
        hparams <dict> with transformer params
        """
        hparams_ = hparams.copy()
        self.dropout = nn.Dropout(hparams['p_dropout'])
        self.p_dropout_out = hparams_.pop('p_dropout_out', 0.0)
        if self.p_dropout_out > 0.0:
            self.dropout_out = nn.Dropout(self.p_dropout_out)

        self.apply_norm_out = hparams_.pop('apply_norm_out', True)
        if self.apply_norm_out:
            self.norm_out = LayerNorm(hparams['d_model'], use_bias=False)

        n_layers = hparams_.pop('n_layers')
        init_weight_method = hparams_.pop('init_weight_method', 'gpt2')
        self.layers = nn.ModuleList()
        layer_scale_init = hparams['layer_scale_init']
        layer_scale_decay = hparams_.pop('layer_scale_decay')
        for i in range(n_layers):
            hparams_['layer_scale_init'] = layer_scale_init
            self.layers.append(TransformerBlock(**hparams_))
            layer_scale_init *= layer_scale_decay

        self.add_position_embeddings = False
        if hparams['pos_emb'].get("name", None) == 'learnable_v2':
            self.add_position_embeddings = True
            self.position_embeddings = nn.Embedding(
                hparams['max_length_causal_mask'], hparams['d_model']
            )
        elif hparams['pos_emb'].get("name", None) == 'learnable':
            # raise warning
            logging.warning("Use learnable_v2 instead of learnable for pos emb, unless using old checkpoints.")

        if init_weight_method == 'gpt2':
            self.apply(self._init_weights_gpt2)
            for pn, p in self.named_parameters():
                if 'o_net' in pn and pn.endswith('weight'):
                    torch.nn.init.normal_(
                        p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
        

    def reset_cache(self, use_cache=False):
        for layer in self.layers:
            layer.reset_cache(use_cache)

    def _init_weights_gpt2(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, x_mask, cond, cond_mask, dump_attention=False,
                attn_prior=None, multi_encoder_mapping=None):
        """
        x <torch tensor> (B, T1, C):
        x_mask <bool mask> (B, T1): True where ignoring is required
        cond <torch tensor> (B, Tc, C) or list of such tensors (from different encoders)
        cond_mask <bool mask> (B, T2): True where ignoring is required or list of such tensors (from different encoders)
        output <torch tensor> (B, T1, C)
        multi_encoder_mapping <list> <int>: None or Same size as n_layers, value indicates which cond input to use for this layer
        """
        if self.add_position_embeddings:
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            pos_emb = self.position_embeddings(positions)
            x = x + pos_emb

        attn_probabilities = []
        x = self.dropout(x)
        for idx, layer in enumerate(self.layers):
            if multi_encoder_mapping is not None:
                if multi_encoder_mapping[idx] is None:
                    # No conditioning for this layer
                    _cond, _cond_mask, _attn_prior = None, None, None
                else:
                    _cond = cond[multi_encoder_mapping[idx]]
                    _cond_mask = cond_mask[multi_encoder_mapping[idx]]
                    _attn_prior = None if attn_prior is None else attn_prior[multi_encoder_mapping[idx]]
            else:
                _cond = cond
                _cond_mask = cond_mask
                _attn_prior = attn_prior
            out_dict = layer(
                x, x_mask, _cond, _cond_mask, dump_attention=dump_attention,
                attn_prior=_attn_prior, idx=idx)
            x = out_dict['output']
            attn_prob = out_dict['attn_probabilities']
            attn_probabilities.append(attn_prob)

        if self.apply_norm_out:
            x = self.norm_out(x)

        if self.p_dropout_out > 0.0:
            x = self.dropout(x)

        return {
            'output': x,
            'attn_probabilities': attn_probabilities
        }

    @torch.no_grad()
    def infer(self, x, cond, x_mask, cond_mask, dump_attention=False,
              attn_prior=None):
        return self.forward(
            x, x_mask, cond, cond_mask, dump_attention=dump_attention,
            attn_prior=attn_prior)
