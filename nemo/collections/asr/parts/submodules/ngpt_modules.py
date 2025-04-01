# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import torch.nn as nn
from einops import rearrange, repeat
from torch import einsum
from flash_attn import flash_attn_func
from nemo.collections.asr.modules.transformer.transformer_modules import FixedPositionalEncoding

def justnorm(x, fp32: bool = False, idim: int = -1):
    if fp32:
        dtype = x.dtype
        x = x.float()
        res = (x / x.norm(p=2, dim=idim, keepdim=True)).to(dtype=dtype)
    else:
        res = x / x.norm(p=2, dim=idim, keepdim=True)
    return res

def justnorm_fp32(x, idim: int = -1):
    return justnorm(x, idim=idim, fp32=True)

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

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


def get_sinusoidal_embeddings(
    seq_len: int,
    dim: int,
    theta: float = 10000.0,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
):
    """
    Generates sinusoidal embeddings (complex numbers represented as real/imag pairs)
    for rotary position embedding.

    Args:
        seq_len: Length of the sequence (T).
        dim: Dimension of the embeddings (D_rotary <= D). Must be even.
        theta: Base value for frequency calculation.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        A tensor of shape (seq_len, dim) containing the embeddings.
    """
    assert dim % 2 == 0, "Dimension must be even."

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype, device=device)[: (dim // 2)] / dim))

    seq_positions = torch.arange(seq_len, device=device, dtype=dtype)

    freqs = einsum("i , j -> i j", seq_positions, freqs)

    emb = repeat(freqs, '... n -> ... (n r)', r=2)

    return emb


def apply_rotary_position_embeddings(freqs: torch.Tensor, t: torch.Tensor, scale: float = 1.0):
    """
    Applies pre-computed rotary embeddings to a tensor. Handles inputs like BxNxTxD.

    Args:
        t: The input tensor (e.g., query or key) of shape (..., seq_len, dim) = (..., T, D).
        freqs: Pre-computed sinusoidal embeddings from get_sinusoidal_embeddings,
               shape (seq_len, dim_rotary) = (T, D_rotary) where D_rotary <= D.
        scale: Optional scaling factor.

    Returns:
        The tensor with rotary embeddings applied, shape (..., T, D).
    """
    rot_dim = freqs.shape[-1]  # D_rotary
    seq_len = freqs.shape[-2]  # T

    # Ensure freqs broadcast correctly to t's shape (..., T, D_rotary)
    # Add leading dimensions to freqs to match t's rank, except for the last two.
    leading_dims = t.shape[:-2]
    freqs_broadcast = freqs.view((1,) * len(leading_dims) + freqs.shape)

    t_dtype = t.dtype
    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]

    t_rotated = (t_rot * freqs_broadcast.cos() * scale) + (rotate_half(t_rot) * freqs_broadcast.sin() * scale)

    out = torch.cat((t_rotated, t_pass), dim=-1)

    return out.type(t_dtype)

# def n_gpt_norm()

class AttentionBlock(nn.Module):
    # Basic block of the nGPT decoder with cross-attention mechanism

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.att_c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = config.base_scale
        self.attn_alpha = torch.nn.Parameter(
            self.attn_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32)
        )

        self.sqk_init_value = 1.0
        self.sqk_init_scaling = config.base_scale
        # trainable scaling factors for each head
        self.sqk = torch.nn.Parameter(self.sqk_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32))

        self.positional_embeddings = FixedPositionalEncoding(config.n_embd, max_sequence_length=1024)

    def normalize_matrices(self):
        
        self.query.weight.data.copy_(justnorm_fp32(self.query.weight.data, 1))
        self.key.weight.data.copy_(justnorm_fp32(self.key.weight.data, 1))
        self.value.weight.data.copy_(justnorm_fp32(self.value.weight.data, 1))
        self.att_c_proj.weight.data.copy_(justnorm_fp32(self.att_c_proj.weight.data, 0))
        

    def forward(self, query, key, value, mask=False):
        # Query from Decoder, Key, Value from Encoder
        # order: SA -> Norm -> Proj -> Res 
        B, Tq, C = query.size()
        _, Tk, _ = key.size()

        hin = query

        # Query, Key, Value
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)

        sinusoidal_pos_q = get_sinusoidal_embeddings(Tq, self.config.n_embd // self.config.n_heads, device=q.device)
        sinusoidal_pos_k = get_sinusoidal_embeddings(Tk, self.config.n_embd // self.config.n_heads, device=k.device)
        
        positional_ids = torch.arange(Tq, device=q.device, dtype=torch.long).unsqueeze(0).repeat(B, 1)
        positional_ids_k = torch.arange(Tk, device=k.device, dtype=torch.long).unsqueeze(0).repeat(B, 1)
        # import pdb; pdb.set_trace()
        q = self.positional_embeddings(positional_ids) + q
        k = self.positional_embeddings(positional_ids_k) + k
        # Divide the embeddings into n_heads
        q = q.view(B, Tq, self.config.n_heads, self.config.n_embd // self.config.n_heads)
        k = k.view(B, Tk, self.config.n_heads, self.config.n_embd // self.config.n_heads)
        v = v.view(B, Tk, self.config.n_heads, self.config.n_embd // self.config.n_heads)

        # Apply the sinusoidal positional embeddings

        # q = apply_rotary_position_embeddings(sinusoidal_pos_q, q.transpose(1, 2)).transpose(2, 1) # BxHxTxH_DIM -> BxTxHxH_DIM 
        # k = apply_rotary_position_embeddings(sinusoidal_pos_k, k.transpose(1, 2)).transpose(2, 1)

        # Scale the query and key
        sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
            1, 1, self.config.n_heads, self.config.n_embd // self.config.n_heads
        )

        q = sqk * justnorm(q)
        k = sqk * justnorm(k)

        sqrt_head_dim = (self.config.n_embd / self.config.n_heads) ** 0.5

        softmax_scale = sqrt_head_dim
        y = flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=mask,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
        )
        y = y.to(dtype=q.dtype)
        y = y.contiguous().view(B, Tq, self.config.n_embd)

        h_att = self.att_c_proj(y)

        lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
        lr = torch.abs(lr)

        A_norm = justnorm(hin)  # normally, normalization is not needed
        B_norm = justnorm(h_att)

        # Residual connection
        # res = (1.0 - lr) * A_norm + lr * B_norm
        res = A_norm + lr * (B_norm - A_norm)
        h = justnorm(res)

        return h
    
    


class MLPBlock(nn.Module):
    # MLP block of the nGPT decoder

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=config.bias)
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)


        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = config.base_scale
        self.mlp_alpha = torch.nn.Parameter(
            self.mlp_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32)
        )

        self.suv_init_value = 1.0
        self.suv_init_scaling = 1.0
        self.suv = torch.nn.Parameter(
            self.suv_init_scaling * torch.ones(2 * 4 * config.n_embd, dtype=torch.float32)
        )

    def normalize_matrices(self):
        
        self.c_fc.weight.data.copy_(justnorm_fp32(self.c_fc.weight.data, 1))
        self.mlp_c_proj.weight.data.copy_(justnorm_fp32(self.mlp_c_proj.weight.data, 0))

    def forward(self, hin):
        # FF (hin is already normalized)
        uv = self.c_fc(hin)
        
        #scaling
        suv = self.suv * ((self.suv_init_value / self.suv_init_scaling) * (self.config.n_embd**0.5))
        uv = suv * uv

        # Split the embeddings into u and v
        u, v = torch.chunk(uv, 2, dim=-1)
        x_mlp = u * self.silu(v)

        # final projection
        h_mlp = self.mlp_c_proj(x_mlp)

        # scaling
        lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
        lr = torch.abs(lr)

        A_norm = justnorm(hin)  # normally, normalization is not needed
        B_norm = justnorm(h_mlp)

        # Residual connection
        # res = (1.0 - lr) * A_norm + lr * B_norm
        res = A_norm + lr * (B_norm - A_norm)
        h = justnorm(res)

        return h
