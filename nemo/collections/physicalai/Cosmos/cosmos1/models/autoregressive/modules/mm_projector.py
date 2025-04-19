# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multimodal projector to connect vision encoder / tokenizer with the LLM."""

from typing import Any, Optional

import torch
import torch.nn as nn


class DownSampleBlock(nn.Module):
    """Downsample block."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Performs the forward pass of the downsample block.

        Args:
            x (torch.Tensor): The input tensor from ViT's output of a sequence of embeddings.
            Shape: (b, seq_len, c).

        Returns:
            torch.Tensor: The output tensor. Shape: (b, seq_len/4, c*4).
        """
        vit_embeds = x
        # Get h and w as the sqrt of seq length. This assumes that the input is square-shaped.
        h = w = int(vit_embeds.shape[1] ** 0.5)
        b = vit_embeds.shape[0]
        vit_embeds = vit_embeds.reshape(b, h, w, -1)
        vit_embeds = self.flat_square(vit_embeds)
        vit_embeds = vit_embeds.reshape(b, -1, vit_embeds.shape[-1])
        return vit_embeds

    def flat_square(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs spatial downsampling while increasing the number of channels.

        Args:
            x (torch.Tensor): The input tensor reshaped to a 2D grid.
            Shape: (b, h, w, c)

        Returns:
            torch.Tensor: The output tensor after the spatial downsampling.
            Shape: (b, h/2, w/2, c*4)
        """
        b, h, w, c = x.size()
        # If w or h is odd, pad a column or a row of zeros.
        if h % 2 == 1:
            x = torch.concat([x, torch.zeros((b, 1, w, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
            b, h, w, c = x.size()
        if w % 2 == 1:
            x = torch.concat([x, torch.zeros((b, h, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
            b, h, w, c = x.size()
        # 2x spatial downsampling, 4x channel increasing.
        x = x.view(b, h, int(w / 2), int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, int(h / 2), int(w / 2), int(c * 4))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x


class MultimodalProjector(nn.Module):
    """Multimodal projector."""

    def __init__(
        self,
        mm_projector_type: str,
        in_dim: int,
        out_dim: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        if mm_projector_type == "identity":
            self.projector = nn.Identity()
        elif mm_projector_type == "linear":
            self.projector = nn.Linear(in_dim, out_dim)
        elif mm_projector_type == "mlp":
            self.projector = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        elif mm_projector_type == "mlp_downsample":
            self.projector = nn.Sequential(
                DownSampleBlock(),
                nn.LayerNorm(in_dim * 4),
                nn.Linear(in_dim * 4, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
        else:
            raise ValueError(f"Unknown projector type: {mm_projector_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)
