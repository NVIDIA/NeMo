# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from einops import rearrange
from torch import einsum, nn

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']


# Inverse dim formula to find dim based on number of rotations
def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))  # 21


# Find dim range bounds based on rotations
def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Position Embedding from https://arxiv.org/abs/2104.09864.
    """

    def __init__(self, dim: int, seq_len_interpolation_factor: int = None, base: int = 10000,
                 pretrain_max_positional_embeddings: int = 2048, extrapolation_factor: int = 1, attn_factor: int = 1,
                 beta_fast: int = 32, beta_slow: int = 1,
                 max_position_embeddings: int = 2048,
                 use_yarn: bool = False):
        """
        Args:

            dim (int): rotary embedding dimension
            seq_len_interpolation_factor (int): if not None, discrete positions will be interpolated
            by this factor via the trick in https://arxiv.org/abs/2306.15595.
        """
        super().__init__()
        self.use_yarn = use_yarn
        self.base = base
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        if self.use_yarn:
            self.seq_len_interpolation_factor = seq_len_interpolation_factor  # scale
            self.pretrain_max_positional_embeddings = pretrain_max_positional_embeddings
            self.extrapolation_factor = extrapolation_factor
            self.attn_factor = attn_factor
            self.beta_fast = beta_fast
            self.beta_slow = beta_slow
            self.dim = dim
            pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
            inv_freq_extrapolation = 1.0 / pos_freqs
            inv_freq_interpolation = 1.0 / (self.seq_len_interpolation_factor * pos_freqs)

            low, high = find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base,
                                              self.pretrain_max_positional_embeddings)
            inv_freq_mask = (1 - linear_ramp_mask(low, high, self.dim // 2).float()) * self.extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
            inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

            self.register_buffer("inv_freq", inv_freq)
            self.mscale = float(
                get_mscale(self.seq_len_interpolation_factor) * self.attn_factor)  # Get n-d magnitude scaling corrected for interpolation

            self.max_seq_len_cached = max_position_embeddings
            t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer('emb', emb)

        else:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, offset=0):
        if self.use_yarn:
            if max_seq_len > self.max_seq_len_cached:
                self.max_seq_len_cached = max_seq_len

                t = torch.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype).to(self.inv_freq.device)
                freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
                # Different from paper, but it uses a different permutation in order to obtain the same calculation
                emb = torch.cat((freqs, freqs), dim=-1).to(self.inv_freq.device)
                self.register_buffer('emb', emb)
            return rearrange(self.emb, 'n d -> n 1 1 d')
        
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        if self.seq_len_interpolation_factor is not None:
            seq = seq.type_as(self.inv_freq)
            seq *= 1 / self.seq_len_interpolation_factor
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        return rearrange(emb, 'n d -> n 1 1 d')


def _rotate_half(x):
    """
    change sign so the last dimension
    [A, B, C, D] -> [-C, -D, A, B]
    """
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)
