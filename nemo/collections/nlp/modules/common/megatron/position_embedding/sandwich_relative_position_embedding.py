# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import torch.nn as nn
from torch.nn import functional as F

__all__ = ['SandwitchRelativePositionEmbedding']


class SandwitchRelativePositionEmbedding(torch.nn.Module):
    """
    Receptive Field Alignment Enables Transformer Length Extrapolation
    Based on https://arxiv.org/abs/2212.10356
    """

    def __init__(self, num_attention_heads, hidden_size):
        """
        Args:
            num_attention_heads: Number of attention heads
            hidden_size: Hidden size per attention head
        """
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

    def forward(self, query_seq_length, key_seq_length):
        context_position = torch.arange(query_seq_length, dtype=torch.long, device=torch.cuda.current_device())[
            :, None
        ]
        memory_position = torch.arange(key_seq_length, dtype=torch.long, device=torch.cuda.current_device())[None, :]
        relative_position = memory_position - context_position  # shape (query_seq_length, key_seq_length)

        inv_freq = 1.0 / (
            10000 ** (2 * torch.arange(1, self.hidden_size / 2, device=torch.cuda.current_device()) / self.hidden_size)
        )

        _bias = torch.sum(relative_position[:, :, None].repeat(1, 1, len(inv_freq)) * inv_freq, axis=2)
        bias = _bias.repeat(self.num_attention_heads, 1, 1)

        _bias_scales = torch.arange(1, self.num_attention_heads + 1, 1, device=torch.cuda.current_device())
        bias_scales = torch.stack(
            list(
                map(
                    lambda x, y: x * y,
                    _bias_scales,
                    torch.ones(
                        self.num_attention_heads, query_seq_length, key_seq_length, device=torch.cuda.current_device()
                    ),
                )
            )
        )
        scaled_bias = (bias - self.hidden_size / 2) / (bias_scales * 8 / self.num_attention_heads)
        
        return scaled_bias.unsqueeze(0)
