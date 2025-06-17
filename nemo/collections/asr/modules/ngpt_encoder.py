# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.distributed
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging
INF_VAL = 10000.0

try:
    from flash_attn import flash_attn_func
except ImportError:

    def flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=1.0,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
    ):
        """Quick and dirty implementation for prototyping."""
        return nn.functional.softmax(q @ (k * softmax_scale).transpose(2, 3), dim=-1) @ v


__all__ = ['NGPTEncoder']


class NGPTEncoder(NeuralModule, Exportable, AccessMixin):
    """
    Transformer encoder based on nGPT for ASR.
    Based on this paper:
    'nGPT: Normalized Transformer with Representation Learning on the Hypersphere' by Ilya Loshchilov et al.
    https://github.com/NVIDIA/ngpt
    """

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        dev = next(self.parameters()).device
        input_example = torch.randn(max_batch, self._feat_in, max_dim, device=dev)
        input_example_length = torch.randint(max_dim // 4, max_dim, (max_batch,), device=dev, dtype=torch.int64)
        all_input_example = tuple([input_example, input_example_length])
        return all_input_example

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def input_types_for_export(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types_for_export(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def disabled_deployment_input_names(self):
        return set()

    @property
    def disabled_deployment_output_names(self):
        return set()

    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        base_scale: float = 1 / (1024**0.5),  # 1/sqrt(d_model)
        n_heads=4,
        causal_downsampling=False,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_chunking_factor=1,
        subsampling_conv_channels=256,
        use_bias=False,
        dropout=0.1,
        use_nGPT=True,
    ):
        super().__init__()
        self._feat_out = d_model
        if subsampling == "ngpt-frame-stack":
            self.pre_encode = NGPTStackingSubsampling(
                subsampling_factor=subsampling_factor,
                feat_in=feat_in,
                feat_out=d_model,
                use_bias=use_bias,
                base_scale=base_scale,
            )
        else:  # temporary back-compat with 1st expts
            self.pre_encode = ConvSubsampling(
                subsampling=subsampling,
                subsampling_factor=subsampling_factor,
                feat_in=feat_in,
                feat_out=d_model,
                conv_channels=subsampling_conv_channels,
                subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
                activation=nn.ReLU(True),
                is_causal=causal_downsampling,
            )
        self.ngpt = GPT(
            config=GPTConfig(
                n_layer=n_layers,
                n_head=n_heads,
                n_embd=d_model,
                base_scale=base_scale,
                use_nGPT=use_nGPT,
                dropout=dropout,
                bias=use_bias,
            )
        )

    def forward_for_export(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        raise NotImplementedError()

    def streaming_post_process(self, rets, keep_all_outputs=True):
        raise NotImplementedError()
    def _create_masks(self, att_context_size, padding_length, max_audio_length, offset, device):
        
        att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool, device=device)


        if att_context_size[0] >= 0:
            att_mask = att_mask.triu(diagonal=-att_context_size[0])
        if att_context_size[1] >= 0:
            att_mask = att_mask.tril(diagonal=att_context_size[1])
        
        # pad_mask is the masking to be used to ignore paddings
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)

        if offset is not None:
            pad_mask_off = torch.arange(0, max_audio_length, device=device).expand(
                padding_length.size(0), -1
            ) >= offset.unsqueeze(-1)
            pad_mask = pad_mask_off.logical_and(pad_mask)

        if att_mask is not None:
            # pad_mask_for_att_mask is the mask which helps to ignore paddings
            pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
            pad_mask_for_att_mask = torch.logical_and(pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2))
            # att_mask is the masking to be used by the MHA layers to ignore the tokens not supposed to be visible
            att_mask = att_mask[:, :max_audio_length, :max_audio_length]
            # paddings should also get ignored, so pad_mask_for_att_mask is used to ignore their corresponding scores
            att_mask = torch.logical_and(pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device))
            att_mask = ~att_mask

        pad_mask = ~pad_mask
        return pad_mask, att_mask
    


    @typecheck()
    def forward(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        return self.forward_internal(
            audio_signal,
            length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )

    def forward_internal(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
            )

        audio_signal = audio_signal.transpose(1, 2)
        x, length = self.pre_encode(x=audio_signal, lengths=length)
        padding_length = length
        pad_mask, att_mask = self._create_masks(
            att_context_size=[-1,-1],
            padding_length=padding_length,
            max_audio_length=x.size(1),
            offset=None,
            device=x.device,
        )
        x = self.ngpt(x, mask=att_mask)
        x = x.transpose(1, 2)

        return x, length

    def normalize_matrices(self):
        if hasattr(self.pre_encode, "normalize_matrices"):
            self.pre_encode.normalize_matrices()
        self.ngpt.normalize_matrices()


def justnorm(x, fp32: bool = False, idim: int = -1):
    orig_dtype = x.dtype
    eps = torch.finfo(torch.float32 if fp32 else orig_dtype).eps
    
    if fp32:
        x = x.float()
        norm = x.norm(p=2, dim=idim, keepdim=True)
        res = (x / (norm + eps)).to(dtype=orig_dtype)
    else:
        norm = x.norm(p=2, dim=idim, keepdim=True)
        res = x / (norm + eps)

    return res


def justnorm_fp32(x, idim: int = -1):
    return justnorm(x, idim=idim, fp32=True)


class Block(nn.Module):

    def __init__(self, config, rotary_emb):
        super().__init__()
        self.config = config

        self.rotary_emb = rotary_emb
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.att_c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.rotary_emb = RotaryEmbedding(dim =config.n_embd // config.n_head)
        self.c_fc = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=config.bias)
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        if config.use_nGPT == 0:
            self.rmsnorm_att = RMSNorm(config.n_embd)
            self.rmsnorm_mlp = RMSNorm(config.n_embd)

        if config.use_nGPT == 1:
            self.attn_alpha_init_value = 0.05
            self.attn_alpha_init_scaling = config.base_scale
            self.attn_alpha = torch.nn.Parameter(
                self.attn_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32)
            )

            self.mlp_alpha_init_value = 0.05
            self.mlp_alpha_init_scaling = config.base_scale
            self.mlp_alpha = torch.nn.Parameter(
                self.mlp_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32)
            )

            self.sqk_init_value = 1.0
            self.sqk_init_scaling = config.base_scale
            self.sqk = torch.nn.Parameter(self.sqk_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32))

            self.suv_init_value = 1.0
            self.suv_init_scaling = 1.0
            self.suv = torch.nn.Parameter(
                self.suv_init_scaling * torch.ones(2 * 4 * config.n_embd, dtype=torch.float32)
            )

    def forward(self, h, mask):
        B, T, C = h.size()

        hin = h
        if self.config.use_nGPT == 0:
            hin = self.rmsnorm_att(h)

        q = self.query(hin)
        k = self.key(hin)
        v = self.value(hin)

        q = q.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        k = k.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        v = v.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
  
        q = self.rotary_emb.rotate_queries_or_keys(q.transpose(2, 1))
        k = self.rotary_emb.rotate_queries_or_keys(k.transpose(2, 1))
        q = q.transpose(2, 1)
        k = k.transpose(2, 1)
        B, T, H, D = q.shape

        # Apply nGPT specific normalization if enabled
        if self.config.use_nGPT == 1:
            sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
                1, 1, self.config.n_head, self.config.n_embd // self.config.n_head
            )
            q = sqk * justnorm(q)
            k = sqk * justnorm(k)

        # Compute attention scaling factor
        sqrt_head_dim = (self.config.n_embd / self.config.n_head) ** 0.5
        softmax_scale = sqrt_head_dim if self.config.use_nGPT == 1 else 1.0 / sqrt_head_dim

        # Reshape tensors for attention computation
        q_ = q.permute(0, 2, 1, 3)  # (B, H, T, D)
        k_ = k.permute(0, 2, 1, 3)  # (B, H, T, D)
        v_ = v.permute(0, 2, 1, 3)  # (B, H, T, D)

        # Compute attention scores
        scores = torch.matmul(q_, k_.transpose(-2, -1)) * softmax_scale  # (B, H, T, T)

        # Apply attention mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), -INF_VAL)

        # ---- (4) softmax & (optional) dropout ----
        attn = torch.softmax(scores, dim=-1)          # (B, H, T, T)
        attn = attn.to(v_.dtype)                                    # ⇐ match v_ (BF16)

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), 0.0)  

        out = torch.matmul(attn, v_)                  # (B, H, T, D)

        y = out.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)


        h_att = self.att_c_proj(y)

        if self.config.use_nGPT == 0:
            h = h + h_att
        if self.config.use_nGPT == 1:
            lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = justnorm(h)  # normally, normalization is not needed
            B_norm = justnorm(h_att)

            # res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = justnorm(res)

        hin = h
        if self.config.use_nGPT == 0:
            hin = self.rmsnorm_mlp(h)
        uv = self.c_fc(hin)
        if self.config.use_nGPT == 1:
            suv = self.suv * ((self.suv_init_value / self.suv_init_scaling) * (self.config.n_embd**0.5))
            uv = suv * uv
        u, v = torch.chunk(uv, 2, dim=-1)
        x_mlp = u * self.silu(v)
        h_mlp = self.mlp_c_proj(x_mlp)

        if self.config.use_nGPT == 0:
            h = h + h_mlp
        if self.config.use_nGPT == 1:
            lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = justnorm(h)  # normally, normalization is not needed
            B_norm = justnorm(h_mlp)

            # res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = justnorm(res)

        return h


@dataclass
class GPTConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1024
    base_scale: float = 1.0 / (1024.0**0.5)  # 1 / sqrt(n_embd)
    use_nGPT: int = 0
    dropout: float = 0.0
    bias: bool = False


class RMSNorm(torch.nn.Module):
    def __init__(self, embdim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(embdim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        xnorm = x * torch.rsqrt(norm + self.eps)
        xnorm = xnorm.to(dtype=dtype)
        return xnorm * self.weight


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rotary_emb = RotaryEmbedding(dim=config.n_embd // config.n_head)
        self.transformer = nn.ModuleDict(
            dict(
                # wte=nn.Embedding(config.vocab_size, config.n_embd),
                # drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config, rotary_emb=self.rotary_emb) for il in range(config.n_layer)])
            )
        )
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # *we don't use it becuase in the nGPT paper there was no weight tying of weights*
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=config.base_scale / math.sqrt(2 * config.n_layer))
        # report number of parameters
        logging.info("[nGPT] number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

        if config.use_nGPT == 0:
            self.rmsnorm_f = RMSNorm(config.n_embd)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)

    def forward(self, x, mask=None):

        for idx, block in enumerate(self.transformer.h):
            x = block(x, mask=mask)

        if self.config.use_nGPT == 0:
            x = self.rmsnorm_f(x)

        return x

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logging.info(
            f"[nGPT] num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        logging.info(
            f"[nGPT] num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        use_fused = False  # fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        logging.info(f"[nGPT] using fused AdamW: {use_fused}")
        return optimizer

    def normalize_matrices(self):
        if not self.config.use_nGPT:
            return

        transformer = self.transformer
        module = self

        for layer_idx in range(0, module.config.n_layer):
            block = transformer["h"][layer_idx]

            block.query.weight.data.copy_(justnorm_fp32(block.query.weight.data, 1))  # n_proj, n_embd
            block.key.weight.data.copy_(justnorm_fp32(block.key.weight.data, 1))  # n_proj, n_embd
            block.value.weight.data.copy_(justnorm_fp32(block.value.weight.data, 1))  # n_proj, n_embd
            block.att_c_proj.weight.data.copy_(justnorm_fp32(block.att_c_proj.weight.data, 0))  # n_embd, n_proj

            block.c_fc.weight.data.copy_(justnorm_fp32(block.c_fc.weight.data, 1))  # n_proj, n_embd
            block.mlp_c_proj.weight.data.copy_(justnorm_fp32(block.mlp_c_proj.weight.data, 0))  # n_embd, n_proj


class NGPTHead(NeuralModule):
    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        base_scale: float = 1 / (1024**0.5),
        use_log_softmax: bool = True,
        include_blank: bool = True,
        vocabulary=None,  # ignored, included for compatibility
    ) -> None:
        super().__init__()
        self._num_classes = num_classes
        if include_blank:
            self._num_classes += 1
        self.lm_head = nn.Linear(feat_in, self._num_classes, bias=False)
        self.sz_init_scaling = base_scale
        self.sz = torch.nn.Parameter(self.sz_init_scaling * torch.ones(self._num_classes, dtype=torch.float32))
        self.use_log_softmax = use_log_softmax
        self.vocabulary = vocabulary

    def _init_weights(self):
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.base_scale)

    def normalize_matrices(self):
        self.lm_head.weight.data.copy_(justnorm_fp32(self.lm_head.weight.data, 1))

    def forward(self, encoder_output):
        x = encoder_output.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        logits = self.lm_head(x)
        sz = self.sz * (1.0 / self.sz_init_scaling)
        logits = sz * logits
        if self.use_log_softmax:
            logits = nn.functional.log_softmax(logits, dim=-1)
        return logits

    @property
    def num_classes_with_blank(self):
        return self._num_classes


class NGPTStackingSubsampling(torch.nn.Module):
    """Stacking subsampling which simply stacks consecutive frames to reduce the sampling rate
    Args:
        subsampling_factor (int): The subsampling factor
        feat_in (int): size of the input features
        feat_out (int): size of the output features
    """

    def __init__(
        self,
        subsampling_factor: int,
        feat_in: int,
        feat_out: int,
        use_bias: bool = False,
        base_scale: float = 1 / (1024**0.5),
    ):
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.proj_out = torch.nn.Linear(subsampling_factor * feat_in, feat_out, bias=use_bias)
        self.pad_frame = nn.Parameter(torch.ones(feat_in, dtype=torch.float32))

    def _init_weights(self):
        torch.nn.init.normal_(self.proj_out.weight, mean=0.0, std=self.config.base_scale)
        if self.proj_out.bias is not None:
            torch.nn.init.zeros_(self.proj_out.bias)

    def normalize_matrices(self):
        self.proj_out.weight.data.copy_(justnorm_fp32(self.proj_out.weight.data, 0))

    def forward(self, x, lengths):
        b, t, h = x.size()
        pad_size = (self.subsampling_factor - (t % self.subsampling_factor)) % self.subsampling_factor
        lengths = torch.div(lengths + pad_size, self.subsampling_factor, rounding_mode='floor')

        # Pad and fill padding frames (all-zero) with a learnable padding 'embedding'
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        x[(x == 0).all(dim=-1)] = self.pad_frame

        _, t, _ = x.size()
        x = torch.reshape(x, (b, t // self.subsampling_factor, h * self.subsampling_factor))
        x = justnorm(x)
        x = self.proj_out(x)
        x = justnorm(x)

        return x, lengths
