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

import torch
from einops import rearrange
from torch import einsum, nn

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']


class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Position Embedding from https://arxiv.org/abs/2104.09864.
    """

    def __init__(self, dim: int, max_seq_len: int, scaling_type: str = 'linear', scaling_factor: int = None):
        """
        Args:

            dim (int): rotary embedding dimension
            scaling_type (str): linear or dynamic
            scaling_factor (int): if not None, discrete positions will be interpolated by this factor via the trick in https://arxiv.org/abs/2306.15595.
        """
        super().__init__()
        self.base = 10000
        self.dim = dim
        self.max_position_embedding = max_seq_len
        self.max_seq_len_cached = max_seq_len
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        assert self.scaling_type in ['linear', 'dynamic']
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, offset=0):
        if max_seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = max_seq_len
            
        seq = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device) + offset
        
        if self.scaling_factor is not None:
            if self.scaling_type == 'dynamic' and self.max_seq_len_cached > self.max_position_embedding:
                base = self.base * ((self.scaling_factor * self.max_seq_len_cached / self.max_position_embedding) - (self.scaling_factor - 1)) ** (self.dim / (self.dim-2))
                inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(self.inv_freq.device) / self.dim))
                self.register_buffer("inv_freq", inv_freq)
            elif self.scaling_type == 'linear':
                seq = seq.type_as(self.inv_freq)
                seq *= 1 / self.scaling_factor
                
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
