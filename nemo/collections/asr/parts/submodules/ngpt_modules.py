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
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        out = nn.functional.softmax(q @ (k * softmax_scale).transpose(2, 3), dim=-1) @ v
        return out.permute(0, 2, 1, 3)


def justnorm(x, fp32: bool = False, idim: int = -1):
    if fp32:
        dtype = x.dtype
        x = x.float()
        res = (x / x.norm(p=2, dim=idim, keepdim=True)).to(dtype=dtype)
    else:
        res = x / x.norm(p=2, dim=idim, keepdim=True)
    return res


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

def apply_rotary_position_embeddings(sinusoidal_pos, tensor):
    """Applies rotary embeddings to the given tensor (Q or K)."""
    # Split sinusoidal_pos into sin and cos parts
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)

    # Create rotated tensor
    tensor_rot = torch.stack((-tensor[..., 1::2], tensor[..., ::2]), dim=-1)

    # Reshape and apply RoPE
    tensor_rot = torch.reshape(tensor_rot, tensor.shape[:-1] + (tensor.shape[-1] // 2, 2)) * torch.stack((cos, sin), dim=-1)
    tensor_rot = torch.reshape(tensor_rot, tensor.shape)

    return tensor_rot.to(tensor.dtype)

def get_sinusoidal_embeddings(n_positions, dim, device):
    """Generate sinusoidal positional embeddings."""
    position = torch.arange(n_positions, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / dim))
    sinusoidal_emb = torch.empty((n_positions, dim), device=device)
    sinusoidal_emb[:, 0::2] = torch.sin(position * div_term)
    sinusoidal_emb[:, 1::2] = torch.cos(position * div_term)
    return sinusoidal_emb


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

        

    def forward(self, query, key, value, mask):
        # Query from Decoder, Key, Value from Encoder
        # order: SA -> Norm -> Proj -> Res 
        B, Tq, C = query.size()
        _, Tk, _ = key.size()

        hin = query

        # Query, Key, Value
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        # import ipdb; ipdb.set_trace()

        # Divide the embeddings into n_heads
        q = q.view(B, Tq, self.config.n_heads, self.config.n_embd // self.config.n_heads)
        k = k.view(B, Tk, self.config.n_heads, self.config.n_embd // self.config.n_heads)
        v = v.view(B, Tk, self.config.n_heads, self.config.n_embd // self.config.n_heads)

        # Apply the sinusoidal positional embeddings
        sinusoidal_pos_q = get_sinusoidal_embeddings(Tq, self.config.n_embd // self.config.n_heads, device=q.device)
        sinusoidal_pos_k = get_sinusoidal_embeddings(Tk, self.config.n_embd // self.config.n_heads, device=k.device)
        
        q = apply_rotary_position_embeddings(sinusoidal_pos_q, q.transpose(1, 2)).transpose(2, 1)
        k = apply_rotary_position_embeddings(sinusoidal_pos_k, k.transpose(1, 2)).transpose(2, 1)

        # Scale the query and key
        sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
            1, 1, self.config.n_heads, self.config.n_embd // self.config.n_heads
        )
        # import ipdb; ipdb.set_trace()
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
            causal=False,
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

