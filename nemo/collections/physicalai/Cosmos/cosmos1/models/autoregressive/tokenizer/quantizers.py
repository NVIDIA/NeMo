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

"""Quantizers for discrete image and video tokenization."""

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from cosmos1.models.autoregressive.tokenizer.utils import default, pack_one, round_ste, unpack_one


class FSQuantizer(nn.Module):
    """Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505

    Adapted from: https://github.com/lucidrains/vector-quantize-pytorch/blob/9502a1f447876d53fd37685b226bf28f250dc4a3/
    vector_quantize_pytorch/finite_scalar_quantization.py
    [Copyright (c) 2020 Phil Wang]
    https://github.com/lucidrains/vector-quantize-pytorch/blob/9502a1f447876d53fd37685b226bf28f250dc4a3/LICENSE
    """

    def __init__(
        self,
        levels: list[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        **ignore_kwargs,
    ):
        super().__init__()
        self.dtype = ignore_kwargs.get("dtype", torch.float32)
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat).float()
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices: torch.Tensor, project_out=True) -> torch.Tensor:
        """Inverse of `codes_to_indices`."""
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes.to(self.dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """
        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack_one(z, "b * d")

        assert z.shape[-1] == self.dim, f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)

        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, "b n c d -> b n (c d)")

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")
            indices = unpack_one(indices, ps, "b * c")
            dummy_loss = torch.zeros_like(out.mean(dim=[1, 2, 3], keepdim=True))
        else:
            dummy_loss = torch.zeros_like(out.mean(dim=[1, 2], keepdim=True)).unsqueeze(1)

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        return (indices, out.to(self.dtype), dummy_loss)
