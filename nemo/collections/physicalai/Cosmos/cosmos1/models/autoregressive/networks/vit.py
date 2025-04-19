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

"""
This module implements a Vision Transformer (ViT) with 2D Rotary Position Embeddings,
designed for processing image inputs in vision-language models.

This module follows Mistral's vision encoder implementation (for their Pistral-12B VLM):
https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/vision_encoder.py
"""
from functools import partial
from typing import Any, Callable, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from cosmos1.models.autoregressive.modules.normalization import create_norm
from cosmos1.models.autoregressive.networks.transformer import TransformerBlock
from cosmos1.utils import log


def get_vit_config(model_name: str) -> Mapping[str, Any]:
    """
    Get the ViT configuration for a given model name.
    """
    if model_name == "pixtral-12b-vit":
        # The 400M ViT of Pixtral 12B VLM
        return dict(
            dim=1024,
            num_channels=3,
            image_size=1024,
            patch_size=16,
            rope_theta=10000,
            ffn_hidden_size=4096,
            n_layers=24,
            n_heads=16,
            n_kv_heads=16,
            norm_type="rmsnorm",
            norm_eps=1e-5,
            image_token_id=10,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def precompute_freqs_cis_2d(
    dim: int,
    height: int,
    width: int,
    theta: float,
) -> torch.Tensor:
    """
    Precompute 2D complex tensor for rotary position embedding.

    This function generates a 2D complex tensor used for rotary position embeddings,
    which helps the model understand spatial relationships in the input image.

    Args:
        dim (int): Dimension of the model (typically the hidden size divided by number of heads).
        height (int): Height of the image in patches.
        width (int): Width of the image in patches.
        theta (float): Base value for the angle calculation, controls the frequency range.

    Returns:
        torch.Tensor: 2D complex tensor of shape (height, width, dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    h = torch.arange(height, device=freqs.device)
    w = torch.arange(width, device=freqs.device)

    freqs_h = torch.outer(h, freqs[::2]).float()
    freqs_w = torch.outer(w, freqs[1::2]).float()
    freqs_2d = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, width, 1),
            freqs_w[None, :, :].repeat(height, 1, 1),
        ],
        dim=-1,
    )
    return torch.polar(torch.ones_like(freqs_2d), freqs_2d)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting with input tensor.

    This function ensures that the frequency tensor can be properly broadcast
    with the input tensor during the rotary embedding process.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor from precompute_freqs_cis_2d.
        x (torch.Tensor): Input tensor to be embedded.

    Returns:
        torch.Tensor: Reshaped frequency tensor ready for broadcasting.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim, f"ndim is {ndim} but index is {1}"
    assert freqs_cis.shape == (
        x.shape[1],
        x.shape[-1],
    ), f"freqs_cis shape is {freqs_cis.shape} but x shape is {x.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    *args,
    freqs_cis: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to input tensors.

    This function applies the rotary positional embeddings to the query and key tensors,
    which helps the model understand spatial relationships in the input.

    Args:
        xq (torch.Tensor): Query tensor.
        xk (torch.Tensor): Key tensor.
        freqs_cis (torch.Tensor): Precomputed frequencies from precompute_freqs_cis_2d.
        *args: Variable length argument list (unused).
        **kwargs: Arbitrary keyword arguments (unused).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class VisionTransformer(nn.Module):
    """
    Vision Transformer model for image processing.

    This class implements a Vision Transformer that processes images using a patch-based approach
    and applies transformer layers with rotary position embeddings.

    Args:
        dim (int): Dimension of the model (hidden size).
        num_channels (int): Number of input image channels (e.g., 3 for RGB).
        patch_size (int): Size of each image patch (e.g., 16x16 pixels).
        n_layers (int): Number of transformer layers.
        n_heads (int): Number of attention heads.
        ffn_hidden_size (int): Hidden size of the feed-forward network in transformer blocks.
        norm_type (str): Type of normalization to use (e.g., "rmsnorm").
        norm_eps (float): Epsilon value for normalization layers.
        image_size (int): Size of the input image (assumed square).
        rope_theta (float): Base value for rotary position embedding calculation.
        attention_dropout (float): Dropout rate for attention layers.
        hidden_dropout (float): Dropout rate for hidden layers.
        image_token_id (int): Token ID for the image token (if present).
    """

    def __init__(
        self,
        dim: int = 1024,
        num_channels: int = 3,
        patch_size: int = 16,
        n_layers: int = 24,
        n_heads: int = 16,
        n_kv_heads: int = None,
        ffn_hidden_size: int = 4096,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
        image_size: int = 1024,
        rope_theta: float = 1000000.0,
        image_token_id: int = None,
    ):
        super().__init__()
        self.patch_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.ln_pre = create_norm(norm_type=norm_type, dim=dim, eps=norm_eps)
        if n_kv_heads is None:
            n_kv_heads = n_heads
        layer_args = dict(
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dim=dim,
            use_qk_normalization=False,
            max_seq_len=None,
            max_batch_size=None,
            ffn_hidden_size=ffn_hidden_size,
            norm_type=norm_type,
            norm_eps=norm_eps,
            causal_mask=False,  # Full attention in ViT
            head_dim=None,
            insert_cross_attn=False,
            attn_type="full",
        )

        self.transformer = VisionTransformerBlocks(n_layers=n_layers, args=layer_args)

        head_dim = dim // n_heads
        assert head_dim % 2 == 0, "ROPE requires even head_dim"

        self.dim = dim
        self.n_heads = n_heads
        self.max_patches_per_side = image_size // patch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.rope_theta = rope_theta
        self._freqs_cis: Optional[torch.Tensor] = None
        self.image_token_id = image_token_id

        num_params = self.get_num_params()
        log.debug(f"Number  of model parameters: {round(num_params / 1e6, 3)}M")

    @classmethod
    def build(
        cls,
        config: Mapping[str, Any],
    ) -> "VisionTransformer":
        """
        Create a Vision Transformer from a configuration dictionary.

        This class method creates a Vision Transformer from a configuration dictionary,
        which is typically loaded from a JSON file or other configuration source.

        Args:
            config (Mapping[str, Any]): Configuration dictionary for the Vision Transformer.

        Returns:
            VisionTransformer: Vision Transformer model instance.
        """
        necessary_keys = ["dim", "num_channels", "patch_size", "n_layers", "n_heads", "ffn_hidden_size", "rope_theta"]
        missing_keys = [k for k in necessary_keys if k not in config]
        assert len(missing_keys) == 0, f"Missing keys in config: {missing_keys}"
        return cls(
            **config,
        )

    def expand_in_channels(self, new_in_channels: int):
        """
        Expand the input channels of the patch convolution layer.
        This is useful when the input is non-standard, e.g. a 4-channel image with the last channel as the alpha channel.
        Note that you should only call this method after the weight is loaded.
        """
        assert (
            new_in_channels > self.patch_conv.in_channels
        ), "Cannot expand the input channels of the patch convolution layer to be less than the original number of channels."
        log.debug(
            f"Vision encoder in_channels is {self.patch_conv.in_channels}. But you have specified to be {new_in_channels}. We will change it to {new_in_channels} channels with {new_in_channels - self.patch_conv.in_channels} channels of 0s."
        )
        new_conv = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=self.patch_conv.out_channels,
            kernel_size=self.patch_conv.kernel_size,
            stride=self.patch_conv.stride,
            bias=False,
        )
        new_conv.weight.data[:, : self.patch_conv.in_channels].copy_(self.patch_conv.weight.data)
        new_conv.weight.data[
            :, self.patch_conv.in_channels :
        ].zero_()  # zeroize, such that initially it has no effect to output
        self.patch_conv = new_conv

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        """
        Get or compute the frequency tensor for rotary position embedding.

        This property lazily initializes and caches the frequency tensor used for
        rotary position embeddings, ensuring it's on the correct device.

        Returns:
            torch.Tensor: The frequency tensor for rotary position embeddings.
        """
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis_2d(
                dim=self.dim // self.n_heads,
                height=self.max_patches_per_side,
                width=self.max_patches_per_side,
                theta=self.rope_theta,
            )

        if self._freqs_cis.device != self.device:
            self._freqs_cis = self._freqs_cis.to(device=self.device)

        return self._freqs_cis

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.

        This method processes the input image through the Vision Transformer,
        including patch embedding, position embedding, and transformer layers.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where B is batch size,
                              C is number of channels, and H, W are height and width.

        Returns:
            torch.Tensor: Output features of shape (B, N, D), where N is the number of patches
                          and D is the embedding dimension.
        """

        patch_embeds = self.patch_conv(x)  # (B, D, Hp, Wp)
        _, _, Hp, Wp = patch_embeds.shape  # Patch embeds dim
        patch_embeds = patch_embeds.flatten(2)  # (B, D, Hp*Wp)
        patch_embeds = patch_embeds.transpose(1, 2)  # (B, Hp*Wp, D)
        patch_embeds = self.ln_pre(patch_embeds)  # (B, Hp*Wp, D)
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(Hp),
                torch.arange(Wp),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)

        freqs_cis = self.freqs_cis[positions[:, 0], positions[:, 1]]
        rope = partial(apply_rotary_emb, freqs_cis=freqs_cis)
        out = self.transformer(patch_embeds, rope=rope)

        return out

    def get_num_params(
        self,
    ) -> int:
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params


class VisionTransformerBlocks(nn.Module):
    """
    Vision Transformer Blocks.

    This class implements a stack of Transformer blocks used in the Vision Transformer.

    Args:
        n_layers (int): Number of transformer layers.
        args (Mapping[str, Any]): Arguments for each transformer block, including dimensions,
    """

    def __init__(
        self,
        n_layers: int,
        args: Mapping[str, Any],
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        for layer_id in range(n_layers):
            self.layers.append(
                TransformerBlock(
                    layer_id=layer_id,
                    args=args,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        rope: Callable,
    ) -> torch.Tensor:
        """
        Forward pass through the Vision Transformer Blocks.

        This method applies a series of Transformer blocks to the input tensor,
        using the provided rotary position embedding function.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D), where B is batch size,
                              N is the number of patches, and D is the embedding dimension.
            rope (Callable): Rotary position embedding function to be applied in each layer.

        Returns:
            torch.Tensor: Output tensor after passing through all transformer layers,
                          with the same shape as the input.
        """
        for layer in self.layers:
            x = layer(x, input_pos=None, mask=None, rope=rope)
        return x
