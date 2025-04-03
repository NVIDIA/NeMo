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

from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn
from torchvision import transforms

from cosmos1.models.diffusion.module.blocks import PatchEmbed
from cosmos1.models.diffusion.networks.general_dit import GeneralDIT


class DiffusionDecoderGeneralDIT(GeneralDIT):
    def __init__(
        self,
        *args,
        is_diffusion_decoder: bool = True,
        diffusion_decoder_condition_on_sigma: bool = False,
        diffusion_decoder_condition_on_token: bool = False,
        diffusion_decoder_token_condition_voc_size: int = 64000,
        diffusion_decoder_token_condition_dim: int = 32,
        **kwargs,
    ):
        # diffusion decoder setting
        self.is_diffusion_decoder = is_diffusion_decoder
        self.diffusion_decoder_condition_on_sigma = diffusion_decoder_condition_on_sigma
        self.diffusion_decoder_condition_on_token = diffusion_decoder_condition_on_token
        self.diffusion_decoder_token_condition_voc_size = diffusion_decoder_token_condition_voc_size
        self.diffusion_decoder_token_condition_dim = diffusion_decoder_token_condition_dim
        super().__init__(*args, **kwargs)

    def initialize_weights(self):
        # Initialize transformer layers:
        super().initialize_weights()
        if self.diffusion_decoder_condition_on_token:
            nn.init.constant_(self.token_embedder.weight, 0)

    def build_patch_embed(self):
        (
            concat_padding_mask,
            in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
            is_diffusion_decoder,
            diffusion_decoder_token_condition_dim,
            diffusion_decoder_condition_on_sigma,
        ) = (
            self.concat_padding_mask,
            self.in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
            self.is_diffusion_decoder,
            self.diffusion_decoder_token_condition_dim,
            self.diffusion_decoder_condition_on_sigma,
        )
        in_channels = (
            in_channels + in_channels
            if (is_diffusion_decoder and not self.diffusion_decoder_condition_on_token)
            else in_channels
        )
        in_channels = in_channels + 1 if diffusion_decoder_condition_on_sigma else in_channels
        in_channels = (
            in_channels + self.diffusion_decoder_token_condition_dim
            if self.diffusion_decoder_condition_on_token
            else in_channels
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels

        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
            bias=False,
        )

        if self.diffusion_decoder_condition_on_token:
            self.token_embedder = nn.Embedding(
                self.diffusion_decoder_token_condition_voc_size, self.diffusion_decoder_token_condition_dim
            )

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepares an embedded sequence tensor by applying positional embeddings and handling padding masks.

        Args:
            x_B_C_T_H_W (torch.Tensor): video
            fps (Optional[torch.Tensor]): Frames per second tensor to be used for positional embedding when required.
                                    If None, a default value (`self.base_fps`) will be used.
            padding_mask (Optional[torch.Tensor]): current it is not used

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of shape (B, T, H, W, D) with the embedded sequence.
                - An optional positional embedding tensor, returned only if the positional embedding class
                (`self.pos_emb_cls`) includes 'rope'. Otherwise, None.

        Notes:
            - If `self.concat_padding_mask` is True, a padding mask channel is concatenated to the input tensor.
            - The method of applying positional embeddings depends on the value of `self.pos_emb_cls`.
            - If 'rope' is in `self.pos_emb_cls` (case insensitive), the positional embeddings are generated using
                the `self.pos_embedder` with the shape [T, H, W].
            - If "fps_aware" is in `self.pos_emb_cls`, the positional embeddings are generated using the `self.pos_embedder`
                with the fps tensor.
            - Otherwise, the positional embeddings are generated without considering fps.
        """
        if self.diffusion_decoder_condition_on_token:
            latent_condition = self.token_embedder(latent_condition)
            B, _, T, H, W, _ = latent_condition.shape
            latent_condition = rearrange(latent_condition, "B 1 T H W D -> (B T) (1 D) H W")

            latent_condition = transforms.functional.resize(
                latent_condition, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.BILINEAR
            )
            latent_condition = rearrange(latent_condition, "(B T) D H W -> B D T H W ", B=B, T=T)
        x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, latent_condition], dim=1)
        if self.diffusion_decoder_condition_on_sigma:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, latent_condition_sigma], dim=1)
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb

        if "fps_aware" in self.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, fps=fps)  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]
        return x_B_T_H_W_D, None, extra_pos_emb
