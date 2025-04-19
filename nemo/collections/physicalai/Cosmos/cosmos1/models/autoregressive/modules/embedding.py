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

# pylint: disable=C0115,C0116,C0301

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange, repeat


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def _rotate_half_te(x: torch.Tensor) -> torch.Tensor:
    """
    change sign so the last dimension becomes [-odd, +even].
    Adopted from TransformerEngine.
    Source: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py
    """
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_te(
    t: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary positional embedding tensor to the input tensor.
    Adopted from TransformerEngine.
    Source: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py

    Parameters
    ----------
    t: torch.Tensor
        Input tensor of shape `[b, s, h, d]`, on which
        rotary positional embedding will be applied.
    cos_freqs: torch.Tensor
        Cosine component of rotary positional embedding tensor of shape `[s, 1, 1, d]` and dtype 'float',
    sin_freqs: torch.Tensor
        Sine component of rotary positional embedding tensor of shape `[s, 1, 1, d]` and dtype 'float',
    """
    rot_dim = cos_freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * cos_freqs) + (_rotate_half_te(t) * sin_freqs)
    output = torch.cat((t, t_pass), dim=-1)
    return output


class RotaryPositionEmbedding(torch.nn.Module):
    """
    Rotary Position Embedding module as described in the paper:
    https://arxiv.org/abs/2104.09864

    This module implements rotary positional embeddings, which are used to
    enhance the performance of transformer models.

    Args:
        dim (int): Dimensionality of the input tensor.
        max_position_embeddings (Optional[int]): Maximum position embeddings.
        original_max_position_embeddings (Optional[int]): Original maximum position embeddings.
        rope_theta (Optional[float]): Base for the frequency calculation.
        apply_yarn (Optional[bool]): Whether to apply YaRN (Yet another Rotary).
        scale (Optional[int]): Scaling factor for the frequency calculation.
        extrapolation_factor (Optional[int]): Extrapolation factor for the frequency extension.
        attn_factor (Optional[int]): Attention factor for the frequency calculation.
        beta_fast (Optional[int]): Fast beta value for the YaRN frequency calculation.
        beta_slow (Optional[int]): Slow beta value for the YaRN frequency calculation.
        rope_dim (Optional[str]): Dimensionality of the RoPE. Choices: "1D", "2D", "3D".
        latent_shape (Optional[List[int]]): Shape of the latent tensor for video or image inputs.
        original_latent_shape (Optional[List[int]]): Original shape of the latent tensor for video or image inputs.
        pad_to_multiple_of (Optional[int]): Pad the position embedding to a multiple of this value.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: Optional[int] = None,
        original_max_position_embeddings: Optional[int] = None,
        rope_theta: Optional[float] = 10000.0,
        apply_yarn: Optional[bool] = False,
        scale: Optional[int] = None,
        extrapolation_factor: Optional[int] = 1,
        attn_factor: Optional[int] = 1,
        beta_fast: Optional[int] = 32,
        beta_slow: Optional[int] = 1,
        rope_dim: Optional[str] = "1D",
        latent_shape: Optional[List[int]] = None,
        original_latent_shape: Optional[List[int]] = None,
        pad_to_multiple_of: Optional[int] = None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.rope_theta = rope_theta
        self.apply_yarn = apply_yarn
        self.scale = scale
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = 1.0
        self.rope_dim = rope_dim
        self.latent_shape = latent_shape
        self.original_latent_shape = original_latent_shape
        self.pad_to_multiple_of = pad_to_multiple_of
        self.get_inv_freq(torch.cuda.current_device())

    def get_mscale(self, scale: float = 1.0) -> float:
        """Get the magnitude scaling factor for YaRN."""
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def forward(self, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass for the rotary position embedding.

        Args:
            seq_len (Optional[int]): Length of the sequence.

        Returns:
            torch.Tensor: The computed frequencies for positional embedding.
        """

        if self.apply_yarn and seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
        self.freqs = self.compute_freqs()

        return self.freqs

    def compute_freqs(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the spatial frequencies for the latent tensor."""
        self.seq = torch.arange(self.max_seq_len_cached, dtype=torch.float).cuda()
        if self.rope_dim == "1D":
            emb = torch.einsum("i,j->ij", self.seq, self.inv_freq)

        elif self.rope_dim == "2D":
            H, W = self.latent_shape
            half_emb_h = torch.outer(self.seq[:H], self.spatial_inv_freq)
            half_emb_w = torch.outer(self.seq[:W], self.spatial_inv_freq)
            emb = torch.cat(
                [
                    repeat(half_emb_h, "h d -> h w d", w=W),
                    repeat(half_emb_w, "w d -> h w d", h=H),
                ]
                * 2,
                dim=-1,
            )
            emb = rearrange(emb, "h w d -> (h w) 1 1 d").float()

        elif self.rope_dim == "3D":
            T, H, W = self.latent_shape
            half_emb_t = torch.outer(self.seq[:T], self.temporal_inv_freq)
            half_emb_h = torch.outer(self.seq[:H], self.spatial_inv_freq)
            half_emb_w = torch.outer(self.seq[:W], self.spatial_inv_freq)
            emb = torch.cat(
                [
                    repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                    repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                    repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
                ]
                * 2,
                dim=-1,
            )
            emb = rearrange(emb, "t h w d -> (t h w) 1 1 d").float()
        else:
            raise ValueError(f"Invalid RoPE dimensionality: {self.rope_dim}")
        return emb

    def get_scale_factors(self, inv_freq: torch.Tensor, original_seq_len: int) -> torch.Tensor:
        """Get the scale factors for YaRN."""
        # Calculate the high and low frequency cutoffs for YaRN. Note: `beta_fast` and `beta_slow` are called
        # `high_freq_factor` and `low_freq_factor` in the Llama 3.1 RoPE scaling code.
        high_freq_cutoff = 2 * math.pi * self.beta_fast / original_seq_len
        low_freq_cutoff = 2 * math.pi * self.beta_slow / original_seq_len
        # Obtain a smooth mask that has a value of 0 for low frequencies and 1 for high frequencies, with linear
        # interpolation in between.
        smooth_mask = torch.clamp((inv_freq - low_freq_cutoff) / (high_freq_cutoff - low_freq_cutoff), min=0, max=1)
        # For low frequencies, we scale the frequency by 1/self.scale. For high frequencies, we keep the frequency.
        scale_factors = (1 - smooth_mask) / self.scale + smooth_mask
        return scale_factors

    def get_inv_freq(self, device: torch.device) -> None:
        """Get the inverse frequency."""
        if self.rope_dim == "1D":
            assert self.max_position_embeddings is not None, "Max position embeddings required."
            inv_freq = 1.0 / (
                self.rope_theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
            )
            if self.apply_yarn:
                assert self.original_max_position_embeddings is not None, "Original max position embeddings required."
                assert self.beta_slow is not None, "Beta slow value required."
                assert self.beta_fast is not None, "Beta fast value required."

                scale_factors = self.get_scale_factors(inv_freq, self.original_max_position_embeddings)
                # Apply the scaling factors to inv_freq.
                inv_freq = inv_freq * scale_factors
                # Set the magnitude scaling factor.
                self.mscale = float(self.get_mscale(self.scale) * self.attn_factor)
            self.max_seq_len_cached = self.max_position_embeddings
            self.inv_freq = inv_freq

        elif self.rope_dim == "2D":
            assert self.latent_shape is not None, "Latent shape required."
            dim_h = self.dim // 2
            spatial_inv_freq = 1.0 / (
                self.rope_theta ** torch.arange(0, dim_h, 2, dtype=torch.float32, device=device) / dim_h
            )
            if self.apply_yarn:
                assert self.original_latent_shape is not None, "Original latent shape required."
                assert self.beta_slow is not None, "Beta slow value required."
                assert self.beta_fast is not None, "Beta fast value required."

                scale_factors = self.get_scale_factors(spatial_inv_freq, self.original_latent_shape[0])
                spatial_inv_freq = spatial_inv_freq * scale_factors
                self.mscale = float(self.get_mscale(self.scale) * self.attn_factor)
            self.spatial_inv_freq = spatial_inv_freq
            self.max_seq_len_cached = max(self.latent_shape)

        elif self.rope_dim == "3D":
            assert self.latent_shape is not None, "Latent shape required."
            dim_h = self.dim // 6 * 2
            dim_t = self.dim - 2 * dim_h
            self.dim_spatial_range = torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(device) / dim_h
            spatial_inv_freq = 1.0 / (self.rope_theta**self.dim_spatial_range)
            self.dim_temporal_range = torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(device) / dim_t
            temporal_inv_freq = 1.0 / (self.rope_theta**self.dim_temporal_range)
            if self.apply_yarn:
                assert self.original_latent_shape is not None, "Original latent shape required."
                assert self.beta_slow is not None, "Beta slow value required."
                assert self.beta_fast is not None, "Beta fast value required."
                scale_factors_spatial = self.get_scale_factors(spatial_inv_freq, self.original_latent_shape[1])
                spatial_inv_freq = spatial_inv_freq * scale_factors_spatial
                scale_factors_temporal = self.get_scale_factors(temporal_inv_freq, self.original_latent_shape[0])
                temporal_inv_freq = temporal_inv_freq * scale_factors_temporal
                self.mscale = float(self.get_mscale(self.scale) * self.attn_factor)
            self.spatial_inv_freq = spatial_inv_freq
            self.temporal_inv_freq = temporal_inv_freq
            self.max_seq_len_cached = max(self.latent_shape)
        else:
            raise ValueError(f"Invalid RoPE dimensionality: {self.rope_dim}")

        self.freqs = self.compute_freqs()


class RotaryPositionEmbeddingPytorchV2(RotaryPositionEmbedding):
    """
    Rotary Position Embedding that works in the same way as the TransformerEngine RoPE
    (https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py)

    """

    def __init__(
        self,
        seq_len: int,
        training_type: str = None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        emb = self.create_rope_freqs(seq_len=seq_len, training_type=training_type)
        emb = emb.transpose(0, 1).contiguous()  # [seq, 1, 1, dim] -> [1, seq, 1, dim]
        assert emb.shape[0] == 1 and emb.shape[2] == 1, f"emb shape: {emb.shape}"
        # cos/sin first then dtype conversion for better precision
        self.register_buffer("cos_cached", torch.cos(emb), persistent=False)
        self.register_buffer("sin_cached", torch.sin(emb), persistent=False)

    def create_rope_freqs(self, seq_len: int, training_type: str = None) -> torch.Tensor:
        """
        Create rotary position embedding frequencies.

        Args:
            seq_len (int): Sequence length of a sample.

        Returns:
            torch.Tensor: The computed positional embeddings.
        """
        if self.rope_dim == "1D":
            freqs = super().forward(seq_len=seq_len)
            emb = torch.cat((freqs, freqs), dim=-1)
            emb = emb.reshape(emb.size(0), 1, 1, emb.size(1))

        elif self.rope_dim in ["2D", "3D"]:
            emb = super().forward(seq_len=seq_len)
            if training_type == "text_to_video":
                # since we added <bov> token at the beginning of the video for text2world, we also extend the position embedding by one token in the beginning
                bov_pe = torch.zeros((1, *emb.shape[1:]), device=emb.device)
                emb = torch.cat((bov_pe, emb), dim=0)
        else:
            raise ValueError(f"Invalid RoPE dimensionality: {self.rope_dim}")
        if self.pad_to_multiple_of is not None and emb.shape[0] % self.pad_to_multiple_of != 0:
            # Round up to the nearest multiple of pad_to_multiple_of
            pad_len = self.pad_to_multiple_of - emb.shape[0] % self.pad_to_multiple_of
            emb = torch.cat((emb, torch.zeros((pad_len, *emb.shape[1:]), device=emb.device)), dim=0)

        return emb

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, input_pos: Optional[torch.Tensor] = None, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != self.cos_cached.dtype:
            self.cos_cached = self.cos_cached.to(q.dtype)
            self.sin_cached = self.sin_cached.to(q.dtype)

        cos_emb = self.cos_cached
        sin_emb = self.sin_cached
        if input_pos is not None:
            cos_emb = cos_emb[:, input_pos, :, :]
            sin_emb = sin_emb[:, input_pos, :, :]
        elif seq_len is not None:
            cos_emb = cos_emb[:, :seq_len, :, :]
            sin_emb = sin_emb[:, :seq_len, :, :]
        q = _apply_rotary_pos_emb_te(q, cos_emb, sin_emb)
        k = _apply_rotary_pos_emb_te(k, cos_emb, sin_emb)
        return q, k


class RotaryPositionEmbeddingPytorchV1(RotaryPositionEmbedding):
    """
    Rotary Position Embedding that works in the same way as
    mistral_inference (https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/rope.py)
    or llama3 (https://github.com/meta-llama/llama3/blob/main/llama/model.py)

    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        emb = None
        if self.rope_dim == "1D":
            emb = torch.stack((self.freqs, self.freqs), dim=-1).reshape(*self.freqs.shape[:-1], -1)
        elif self.rope_dim in ["2D", "3D"]:
            emb = rearrange(self.freqs, "s 1 1 d -> s d").float()
        self.register_buffer("cos_cached", (emb.cos() * self.mscale)[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale)[None, :, None, :], persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dimensions of the input tensor."""
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        output = torch.stack((-x2, x1), dim=-1).reshape(*x.shape)
        return output

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, input_pos: Optional[torch.Tensor] = None, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the rotary position embedding.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            input_pos (Optional[torch.Tensor]): Starting position for the sequence.
            seq_len (Optional[int]): Length of the sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
        """
        if self.apply_yarn and seq_len > self.max_seq_len_cached:
            freqs = super().forward(seq_len)
            if self.rope_dim == "1D":
                emb = torch.stack((freqs, freqs), dim=-1).reshape(*freqs.shape[:-1], -1)
            elif self.rope_dim in ["2D", "3D"]:
                emb = rearrange(freqs, "s 1 1 d -> s d").float()
            else:
                raise ValueError(f"Invalid RoPE dimensionality: {self.rope_dim}")
            self.register_buffer(
                "cos_cached", (emb.cos() * self.mscale)[None, :, None, :].to(q.dtype), persistent=False
            )
            self.register_buffer(
                "sin_cached", (emb.sin() * self.mscale)[None, :, None, :].to(q.dtype), persistent=False
            )

        if input_pos is not None:
            cos_cached = self.cos_cached[:, input_pos]
            sin_cached = self.sin_cached[:, input_pos]
        else:
            assert (
                self.cos_cached.shape[1] >= seq_len
            ), f"Invalid sequence length; cos_cached.shape {self.cos_cached.shape}, seq_len {seq_len}."
            cos_cached = self.cos_cached[:, :seq_len, ...]
            sin_cached = self.sin_cached[:, :seq_len, ...]
        xq = q * cos_cached + self.rotate_half(q) * sin_cached
        xk = k * cos_cached + self.rotate_half(k) * sin_cached

        return xq.type_as(q), xk.type_as(k)


class SinCosPosEmbAxisTE(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        latent_shape: Optional[List[int]] = None,
        pad_to_multiple_of: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
        device="cuda",
        **kwargs,
    ):
        """
        Args:
            dim (int): Dimensionality of the input tensor.
            latent_shape (Optional[List[int]]): Shape of the latent tensor for video or image inputs.
            pad_to_multiple_of (Optional[int]): Pad the position embedding to a multiple of this value.
            dtype (torch.dtype): Data type of the position embedding tensor.
        """
        super().__init__()
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self.latent_shape = latent_shape
        T, H, W = latent_shape
        emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, pos=np.arange(H))
        emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, pos=np.arange(W))
        emb_t = get_1d_sincos_pos_embed_from_grid(dim_t, pos=np.arange(T))

        self.register_buffer("pos_emb_h", torch.from_numpy(emb_h).to(dtype=dtype, device=device), persistent=False)
        self.register_buffer("pos_emb_w", torch.from_numpy(emb_w).to(dtype=dtype, device=device), persistent=False)
        self.register_buffer("pos_emb_t", torch.from_numpy(emb_t).to(dtype=dtype, device=device), persistent=False)
        self.pad_to_multiple_of = pad_to_multiple_of

    def forward(
        self,
        training_type: str | None = None,
    ) -> torch.Tensor:
        T, H, W = self.latent_shape
        emb = torch.cat(
            [
                repeat(self.pos_emb_t, "t d-> t h w d", h=H, w=W),
                repeat(self.pos_emb_h, "h d-> t h w d", t=T, w=W),
                repeat(self.pos_emb_w, "w d-> t h w d", t=T, h=H),
            ],
            dim=-1,
        )
        # Flatten the T,H,W dimensions
        emb = rearrange(emb, "t h w d -> (t h w) d")

        if training_type == "text_to_video":
            bov_pe = torch.zeros((1, *emb.shape[1:]), device=emb.device, dtype=emb.dtype)
            emb = torch.cat((bov_pe, emb), dim=0)
        if self.pad_to_multiple_of is not None and emb.shape[0] % self.pad_to_multiple_of != 0:
            pad_len = self.pad_to_multiple_of - emb.shape[0] % self.pad_to_multiple_of
            emb = torch.cat((emb, torch.zeros((pad_len, *emb.shape[1:]), device=emb.device, dtype=emb.dtype)), dim=0)
        seq_len, dim = emb.shape
        emb = emb.reshape(1, seq_len, dim)
        return emb
