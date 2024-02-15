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
from nemo.utils import logging
from typing import Dict, Any
import random

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']


class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Position Embedding from https://arxiv.org/abs/2104.09864.
    """

    def __init__(
        self,
        dim: int,
        seq_len_interpolation_factor: int = None,
        rotary_base: int = 10000,
        base_len: int = None,
        enforce_fp32_pos_idx: bool = False,
        augment_seq: Dict[Any,Any] = None,
        logging_freq: int = 0.01,
    ):
        """
        Args:
            
            dim (int): rotary embedding dimension
            seq_len_interpolation_factor (int): if not None, discrete positions will be interpolated
            by this factor via the trick in https://arxiv.org/abs/2306.15595.
            pretrained_max_position_embeddings (int): pre-trained max_position_embeddings before position interpolation.
        """
        super().__init__()
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.rotary_base = rotary_base
        inv_freq = 1.0 / (self.rotary_base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.base_len = base_len
        self.enforce_fp32_pos_idx = enforce_fp32_pos_idx
        self.augment_seq = augment_seq
        self.logging_freq = logging_freq

        logging.info(f'base_len: {base_len}, rotary_base: {rotary_base}, seq_len_interpolation_factor: {seq_len_interpolation_factor}, enforce_fp32_pos_idx: {enforce_fp32_pos_idx}, augment_seq: {augment_seq}')

    """
        Augments the seq and adjusts its range to base_len
        Args:
            seq (tensor): tensor of positions
            max_seq_len (int): length of this samplw
            Applies stretch and shift augmentations and returns the augmented seq
        """
    def augment(self, seq, max_seq_len):
        current_range = max_seq_len

        target_augmented_length = self.augment_seq.get('target', None)
        augmented_length_range = self.augment_seq.get('range', None)
        if target_augmented_length and augmented_length_range:
            logging.warning(f'target_augmented_length setting of {target_augmented_length} supercedes augmented_length_range of {augmented_length_range}')
        elif augmented_length_range:
            target_augmented_length = random.randint(max(augmented_length_range[0], max_seq_len),augmented_length_range[1])

        if self.augment_seq.get('stretch', False):
            if target_augmented_length:
                max_stretch_factor  = target_augmented_length / current_range
            else:
                max_stretch_factor  = self.base_len * self.seq_len_interpolation_factor / current_range

            stretch_factor = random.random() * max_stretch_factor
            if self.augment_seq.get('discrete', False):
                stretch_factor = int(stretch_factor)
            seq *= stretch_factor
            current_range *= stretch_factor
        
        num_shifts = self.augment_seq.get('num_shifts', None)
        if num_shifts:
            if target_augmented_length:
                total_shift = target_augmented_length - current_range
            else:
                total_shift = self.base_len * self.seq_len_interpolation_factor - current_range

        if self.augment_seq.get('allowed_shift_values', False):
            # provides allowed values for each shift index
            allowed_shift_values = self.augment_seq['allowed_shift_values']
            assert (len(allowed_shift_values) == num_shifts), f'allowed_shift_values length {allowed_shift_values} does not match num_shifts {num_shifts}'
            shifts = torch.zeros(num_shifts, dtype = torch.int)
            for idx, allowed_values in enumerate(allowed_shift_values):
                shifts[idx] = random.choice(allowed_values)
                
        else:
            shifts = torch.rand(num_shifts)
            if augmented_length_range is not None:
                shifts = (augmented_length_range[0] + shifts * (augmented_length_range[1] - augmented_length_range[0]))/ num_shifts
            else:
                shifts = shifts / shifts.sum() * total_shift
            
            if self.augment_seq.get('discrete', False):
                shifts = torch.round(shifts).to(torch.int)

        if self.augment_seq.get('shift_indices', False):
            indices2shift = self.augment_seq['shift_indices']
        else:
            indices2shift = (torch.rand(num_shifts) * max_seq_len).to(torch.int)

        for idx, i in enumerate(indices2shift):
            seq[i:] += shifts[idx]

        if random.random() < self.logging_freq:
            logging.info(f'indices2shift: {indices2shift}, shifts: {shifts}, total shift: {torch.sum(shifts)}')

        return seq
        

    def forward(self, max_seq_len, offset=0, maybe_interpolate=True, maybe_augment=True):
        if random.random() < self.logging_freq:
            logging.info(f'max_seq_len: {max_seq_len}, maybe_interpolate: {maybe_interpolate}, maybe_augment: {maybe_augment}')

        if self.enforce_fp32_pos_idx:
            seq = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=torch.float32) + offset
        else:
            seq = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype) + offset

        if maybe_augment and self.augment_seq and self.augment_seq.get('add_noise', False):
            seq += torch.rand_like(seq) 

        if not maybe_interpolate:
            logging.warning(f'maybe_interpolate set to {maybe_interpolate}')

        if self.base_len is not None and self.seq_len_interpolation_factor is not None and maybe_interpolate:
            if max_seq_len > self.base_len * self.seq_len_interpolation_factor:
                # dynamic linear scaling (length > position we have learned)
                logging.info(f'dynamic interpolation triggered: max_seq_len: {max_seq_len}, base_len: {self.base_len}, seq_len_interpolation_factor: {self.seq_len_interpolation_factor}')
                seq *= 1 / (max_seq_len / self.base_len)
            else:
                # fixed linear scaling
                if maybe_augment and self.augment_seq:
                    seq = self.augment(seq, max_seq_len)

                seq *= 1 / self.seq_len_interpolation_factor
                

        freqs = einsum('i , j -> i j', seq, self.inv_freq)
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
