# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import torch
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple, Union

class ConvolutionLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
        is_causal: bool = False,
    ):
        """
        A convolutional layer that supports causal convolutions with padding. Replaces the standard MLP layer used in the original transformer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution.
            padding (Optional[int]): Padding added to both sides of the input. If None, it's calculated automatically.
            dilation (int): Spacing between kernel elements.
            bias (bool): If True, adds a learnable bias to the output.
            is_causal (bool): If True, uses causal convolution.
        """
        super().__init__()

        padding = 0 if is_causal else padding
        if padding is None:
            if kernel_size % 2 == 0:
                raise ValueError("`kernel_size` must be odd when `padding` is None.")
            else:
                padding = int(dilation * (kernel_size - 1) / 2)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.is_causal = is_causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, signal):
        if self.is_causal:  # TODO: maybe replace with identify rather than keep conditional if in forward
            
            padding = (((self.kernel_size - 1) * self.dilation), 0)
            signal = F.pad(signal, padding)
        conv_signal = self.conv(signal)
        return conv_signal


class PositionwiseConvFF(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        p_dropout: float,
        kernel_size: int = 1,
        bias: bool = False,
        is_causal: bool = True,
        non_linearity: Callable = torch.nn.GELU(approximate="tanh"),
    ):
        """
        Positionwise Convolutional Feed-Forward layer to replace the MLP layer in transformers. 
        
        Module will take the input with d_model hidden state, project it to d_ffn hidden dimension, perform nonlinear
        transformation, and project the state back into d_model hidden dimension. Finally, it applied dropout.

        Args:
            d_model (int): Input and output dimension of the model.
            d_ffn (int): Hidden dimension of the feed-forward network (usually 4 * d_model).
            p_dropout (float): Dropout probability.
            kernel_size (int): Size of the convolving kernel.
            bias (bool): If True, adds a learnable bias to the convolution layers.
            is_causal (bool): If True, uses causal convolution.
            non_linearity (Callable): Activation function to use (default: GELU).
        """
        super().__init__()
        # d_ffn is usually 4*d_model
        self.d_model = d_model
        self.non_linearity = non_linearity

        self.proj = ConvolutionLayer(d_model, d_ffn, bias=bias, kernel_size=kernel_size, is_causal=is_causal)
        self.o_net = ConvolutionLayer(d_ffn, d_model, bias=bias, kernel_size=kernel_size, is_causal=is_causal)
        self.dropout = torch.nn.Dropout(p_dropout)

    def forward(self, x):
        """
        x (B, T, C)
        """
        x = self.non_linearity(self.proj(x.transpose(1, 2)))
        x = self.dropout(self.o_net(x).transpose(1, 2))
        return x


class Attention(torch.nn.Module):
    def __init__(
        self,
        n_heads,
        d_model,
        p_dropout,
        is_causal=True,
        is_self_attention=True,
        d_memory=None,
        deterministic=False,
        max_length_causal_mask=4096,
    ):
        """
        Attention part of transforer. Supports both self-attention and cross-attention depending on is_self_attention
        arg.

        Does DotProductionAttention and additionally dropout inside of the module.
        """
        super().__init__()
        # context conditional attention dims
        if is_self_attention:
            assert d_model % n_heads == 0, "d_model % n_head != 0"
            self.d_head = d_model // n_heads
        else:
            if d_memory is None:
                raise ValueError("d_memory must be provided for cross-attention")
                
            assert d_memory % n_heads == 0, "d_memory % n_head != 0"
            self.d_head = d_memory // n_heads

        self.n_heads = n_heads
        self.d_model = d_model
        self.scale = self.d_head**-0.5
        self.is_causal = is_causal
        self.is_self_attention = is_self_attention
        self.deterministic = deterministic
        self.max_length_causal_mask = max_length_causal_mask

        if is_causal and is_self_attention:
            # ~ 45 seconds mask, 4096 mel frames, 86 frames per second
            # ~ 762 seconds mask, 65536 mel frames, 86 frames per second

            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(max_length_causal_mask, max_length_causal_mask)).view(
                    1, 1, max_length_causal_mask, max_length_causal_mask
                )
                == 0,
            )

        if is_self_attention:
            self.qkv_net = torch.nn.Linear(d_model, 3 * n_heads * self.d_head, bias=False)
            self.o_net = torch.nn.Linear(n_heads * self.d_head, d_model, bias=False)
        else:
            self.q_net = torch.nn.Linear(d_model, n_heads * self.d_head, bias=False)
            self.kv_net = torch.nn.Linear(d_memory, 2 * n_heads * self.d_head, bias=False)
            self.o_net = torch.nn.Linear(n_heads * self.d_head, d_model, bias=False)
        self.dropout = torch.nn.Dropout(p_dropout)
        self.use_cache = False
        self.cache = self._init_cache()
        
    def _init_cache(self) -> Dict[str, Optional[torch.Tensor]]:
        return {
            'is_initialized': False,
            'self_k': None,
            'self_v': None,
            'cross_kv': None,
            'cross_k': None,
            'cross_v': None,
        }

    def reset_cache(self, use_cache: bool = False):
        self.use_cache = use_cache
        self.cache = self._init_cache()

    def attn_naive(self, query: torch.Tensor, query_mask: Optional[torch.Tensor], memory: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None, attn_prior: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.use_cache:
            if self.cache['is_initialized']:
                query = query[:, -1:, :]
                query_mask = query_mask[:, -1:]
            else:
                self.cache['is_initialized'] = True

        B, T, _ = query.shape
        mask = None

        if self.is_self_attention:
            qkv = self.qkv_net(query).reshape(B, T, 3, self.n_heads, self.d_head)
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

        attn_score = bias + torch.matmul(q, k.transpose(2, 3)) * self.scale

        if not self.is_self_attention and memory_mask is not None:
            mask = memory_mask[:, None, None]

        if self.is_self_attention and query_mask is not None:
            mask = query_mask[:, None, :, None]

        if mask is not None:
            # assumes there's at least one mask
            attn_score.masked_fill_(mask, float('-inf'))

        if self.is_self_attention and self.is_causal:
            attn_score.masked_fill_(self.causal_mask[..., :T, :T], float('-inf'))

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

    def forward(self, query, query_mask=None, memory=None, memory_mask=None, attn_prior=None):
        """
        all inputs should be (B, T, C)
        query_mask (T1, T1)
        memory_mask (B, T2)
        attn_prior (T1, T2)

        Returns:
            y: attention module tensor output
            attn_prob: List, returned only in attn_naive
                0th element being the probabilities which are logged during validation
                1st element being the attention scores which are used for ctc loss
        """

        y, attn_prob = self.attn_naive(query, query_mask, memory, memory_mask, attn_prior)

        y = self.dropout(self.o_net(y))

        return y, attn_prob


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        sa_n_heads: int,
        kernel_size: int,
        p_dropout: float,
        has_xattn: bool,
        xa_d_memory: Optional[int] = None,
        xa_n_heads: Optional[int] = None,
        is_causal: bool = True,
        apply_norm_to_cond: bool = True,
        layer_norm_method: str = 'pre',
        deterministic: bool = False,
        max_length_causal_mask: int = 4096,
        conv_non_linearity: Callable = torch.nn.GELU(approximate="tanh"),
    ):
        super().__init__()
        """
        One layer of the Transformer.
        Args:
            d_model <int>: Model dimension
            d_ffn <int>: Feed forward dimension (usually 4*d_model)
            sa_n_heads <int>: Number of attention heads used in self-attention
            kernel_size <int>: Convolution kernel size for FFN
            p_dropout <float>: Dropout probability
            has_xattn <bool>: Whether to use cross attention
            xa_d_memory <int>: Hidden dimenssion for cross attention
            xa_n_heads <int>: Number of attention heads used in cross attention
            is_causal <bool>: Whether to use causal attention
            apply_norm_to_cond <bool>: Whether to apply normalization to conditioning tensor
            layer_norm_method <str>: Layer normalization method
            deterministic <bool>: Whether to use deterministic attention
            max_length_causal_mask <int>: Maximum length of causal mask
            conv_non_linearity <Callable>: Convolution non-linearity
        """
        self.layer_norm_method = layer_norm_method
        self.has_xattn = has_xattn

        self.norm_self = torch.nn.LayerNorm(d_model, bias=False)
        self.self_attention = Attention(
            n_heads=sa_n_heads,
            d_model=d_model,
            p_dropout=p_dropout,
            is_self_attention=True,
            deterministic=deterministic,
            max_length_causal_mask=max_length_causal_mask,
        )

        if self.has_xattn:
            self.apply_norm_to_cond = apply_norm_to_cond
            self.norm_xattn_query = torch.nn.LayerNorm(d_model, bias=False)
            self.cross_attention = Attention(
                n_heads=xa_n_heads,
                d_model=d_model,
                p_dropout=p_dropout,
                is_causal=False,
                is_self_attention=False,
                d_memory=xa_d_memory,
                deterministic=deterministic,
            )

            if self.apply_norm_to_cond:
                self.norm_xattn_memory = torch.nn.LayerNorm(xa_d_memory, bias=False)



        self.norm_pos_ff = torch.nn.LayerNorm(d_model, bias=False)
        self.pos_ff = PositionwiseConvFF(
            d_model, d_ffn, p_dropout, kernel_size=kernel_size, is_causal=is_causal, non_linearity=conv_non_linearity
        )

        self.use_cache = False
        self.cache = self._init_cache()
    
    def _init_cache(self) -> Dict:
        return {
            'self_attn_output': None,
            'cross_attn_output': None,
            'memory': None,
        }
    
    def reset_cache(self, use_cache=False):
        self.use_cache = use_cache
        self.cache = self._init_cache()
        self.self_attention.reset_cache(use_cache)
        if self.has_xattn:
            self.cross_attention.reset_cache(use_cache)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, cond: Optional[torch.Tensor] = None, cond_mask: Optional[torch.Tensor] = None, attn_prior: Optional[torch.Tensor] = None) -> Dict:
        """
        Args:
            x <torch tensor> (B, T1, C): Input tensor
            x_mask <bool mask> (B, T1): True where ignoring is required
            cond <torch tensor> (B, T2, C): Conditioning tensor
            cond_mask <bool mask> (B, T2): True where ignoring is required

        Returns dict with keys
            output <torch tensor> (B, T1, C): Output tensor
            attn_probabilities <dict>: Attention probabilities
        """
        x_mask_inv_float = (~x_mask).to(x.dtype).unsqueeze(-1)
        s_attn_prob = None
        if self.layer_norm_method == 'pre':
            x_, s_attn_prob = self.self_attention(query=self.norm_self(x), query_mask=x_mask)
            if self.use_cache:
                if self.cache['self_attn_output'] is not None:
                    x_ = torch.cat([self.cache['self_attn_output'], x_], dim=1)
                self.cache['self_attn_output'] = x_
            x = x + x_
            x = x * x_mask_inv_float
        elif self.layer_norm_method == 'post':
            x_, s_attn_prob = self.self_attention(query=x, query_mask=x_mask)
            if self.use_cache:
                if self.cache['self_attn_output'] is not None:
                    x_ = torch.cat([self.cache['self_attn_output'], x_], dim=1)
                self.cache['self_attn_output'] = x_
            x = x + x_
            x = self.norm_self(x) * x_mask_inv_float

        x_attn_prob = None
        if self.has_xattn and cond is not None:
            if self.layer_norm_method == 'pre':
                x_normed = self.norm_xattn_query(x)
                if self.use_cache and self.cache['memory'] is not None:
                    memory = self.cache['memory']
                else:
                    memory = self.norm_xattn_memory(cond) if self.apply_norm_to_cond else cond
                    if self.use_cache:
                        self.cache['memory'] = memory

                x_res, x_attn_prob = self.cross_attention(
                    query=x_normed, query_mask=x_mask, memory=memory, memory_mask=cond_mask, attn_prior=attn_prior
                )
                if self.use_cache:
                    if self.cache['cross_attn_output'] is not None:
                        x_res = torch.cat([self.cache['cross_attn_output'], x_res], dim=1)
                    self.cache['cross_attn_output'] = x_res
                x = x + x_res
                x = x * x_mask_inv_float
            elif self.layer_norm_method == 'post':
                x_res, x_attn_prob = self.cross_attention(
                    query=x, query_mask=x_mask, memory=cond, memory_mask=cond_mask, attn_prior=attn_prior
                )
                if self.use_cache:
                    if self.cache['cross_attn_output'] is not None:
                        x_res = torch.cat([self.cache['cross_attn_output'], x_res], dim=1)
                    self.cache['cross_attn_output'] = x_res
                x = (x + x_res) * x_mask_inv_float
                x = self.norm_xattn_query(x)

        # mlp final projection
        if self.layer_norm_method == 'pre':
            x = x + self.pos_ff(self.norm_pos_ff(x))
            x *= x_mask_inv_float
        elif self.layer_norm_method == 'post':
            x = x + self.pos_ff(x)
            x *= x_mask_inv_float
            x = self.norm_pos_ff(x)

        return {
            'output': x,
            'attn_probabilities': {
                'self_attn_probabilities': s_attn_prob, 
                'cross_attn_probabilities': x_attn_prob
            },
        }


class Transformer(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_ffn: int,
        sa_n_heads: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        p_dropout_out: float = 0.0,
        xa_d_memory: Optional[int] = None,
        xa_n_heads: Optional[int] = None,
        has_xattn: bool = False,
        is_causal: bool = True,
        apply_norm_to_cond: bool = True,
        apply_norm_out: bool = False,
        layer_norm_method: str = 'pre',
        deterministic: bool = False,
        max_length_causal_mask: int = 4096,
        use_learnable_pos_emb: bool = False,
        conv_non_linearity: Callable = torch.nn.GELU(approximate="tanh"),
    ):
        """
        Initializes a stack of transformer layers. Can be used for both encoder and decoder.
        Set is_causal is True for autoregressive models. Equivalent to TransformerBlock from Megatron-LM
        Args:
            n_layers <int>: Number of transformer layers
            d_model <int>: Model dimension
            d_ffn <int>: Feed forward dimension (usually 4*d_model)
            sa_n_heads <int>: Number of attention heads used in self-attention
            kernel_size <int>: Convolution kernel size for FFN
            p_dropout <float>: Dropout probability
            p_dropout_out <float>: Dropout probability for output
            xa_d_memory <int>: Hidden dimenssion for cross attention
            xa_n_heads <int>: Number of attention heads used in cross attention
            has_xattn <bool>: Whether to use cross attention
            is_causal <bool>: Whether to use causal attention
            apply_norm_to_cond <bool>: Whether to apply normalization to conditioning tensor
            apply_norm_out <bool>: Whether to apply normalization to output
            layer_norm_method <str>: Layer normalization method
            deterministic <bool>: Whether to use deterministic attention
            max_length_causal_mask <int>: Maximum length of causal mask
            conv_non_linearity <Callable>: Convolution non-linearity
        """
        super().__init__()
        self.dropout = torch.nn.Dropout(p_dropout)
        self.p_dropout_out = p_dropout_out
        if self.p_dropout_out > 0.0:
            self.dropout_out = torch.nn.Dropout(self.p_dropout_out)
        else:
            self.dropout_out = None
        self.apply_norm_out = apply_norm_out
        if self.apply_norm_out:
            self.norm_out = torch.nn.LayerNorm(d_model, bias=False)
        else:
            self.norm_out = None
        self.layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                TransformerLayer(
                    d_model=d_model,
                    d_ffn=d_ffn,
                    sa_n_heads=sa_n_heads,
                    kernel_size=kernel_size,
                    p_dropout=p_dropout,
                    xa_d_memory=xa_d_memory,
                    xa_n_heads=xa_n_heads,
                    has_xattn=has_xattn,
                    is_causal=is_causal,
                    apply_norm_to_cond=apply_norm_to_cond,
                    layer_norm_method=layer_norm_method,
                    deterministic=deterministic,
                    max_length_causal_mask=max_length_causal_mask,
                    conv_non_linearity=conv_non_linearity,
                )
            )

        self.use_learnable_pos_emb = use_learnable_pos_emb
        if self.use_learnable_pos_emb:
            self.position_embeddings = torch.nn.Embedding(max_length_causal_mask, d_model)
        # Apply random uniform init for all layers, except for output layers: The second of the two layers in the MLP
        # and the last linear projection in dot product attention. The output layers are scaled depending on the
        # number of layers
        self.apply(self._init_weights_gpt2)
        for name, param in self.named_parameters():
            if 'o_net' in name and name.endswith('weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def reset_cache(self, use_cache=False):
        for layer in self.layers:
            layer.reset_cache(use_cache)

    @staticmethod
    def _init_weights_gpt2(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    @staticmethod
    def _get_layer_inputs(idx: int, cond: Optional[Union[torch.Tensor, List[torch.Tensor]]], cond_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]], attn_prior: Optional[Union[torch.Tensor, List[torch.Tensor]]], multi_encoder_mapping: Optional[List[Optional[int]]]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if multi_encoder_mapping is not None:
            if multi_encoder_mapping[idx] is None:
                return None, None, None
            else:
                return (
                    cond[multi_encoder_mapping[idx]],
                    cond_mask[multi_encoder_mapping[idx]] if cond_mask is not None else None,
                    attn_prior[multi_encoder_mapping[idx]] if attn_prior is not None else None
                )
        else:
            return cond, cond_mask, attn_prior
        
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, cond: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None, cond_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None, attn_prior: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None, multi_encoder_mapping: Optional[List[Optional[int]]] = None) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Args:
            x <torch tensor> (B, T1, C):
            x_mask <bool mask> (B, T1): True where ignoring is required
            cond <torch tensor> (B, T2, C) or list of such tensors (from different encoders)
            cond_mask <bool mask> (B, T2): True where ignoring is required or list of such tensors (from different
                encoders) output <torch tensor> (B, T1, C)
            multi_encoder_mapping <list> <int>: None or Same size as n_layers, value indicates which cond input to use
                for this layer

        Returns dict with keys:
            output <torch tensor> (B, T1, C): Output tensor
            attn_probabilities <list>: Attention probabilities of each layer
        """
        if self.use_learnable_pos_emb:
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            x = x + self.position_embeddings(positions)

        attn_probabilities = []
        x = self.dropout(x)
        for idx, layer in enumerate(self.layers):
            _cond, _cond_mask, _attn_prior = self._get_layer_inputs(idx, cond, cond_mask, attn_prior, multi_encoder_mapping)
            out_dict = layer(x, x_mask, _cond, _cond_mask, attn_prior=_attn_prior)
            x = out_dict['output']
            attn_probabilities.append(out_dict['attn_probabilities'])

        if self.norm_out is not None:
            x = self.norm_out(x)

        if self.dropout_out is not None:
            x = self.dropout_out(x)

        return {'output': x, 'attn_probabilities': attn_probabilities}