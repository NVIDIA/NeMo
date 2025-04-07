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

from typing import Optional

import torch
from einops import rearrange
from torch import nn

from cosmos1.models.diffusion.conditioner import DataType
from cosmos1.models.diffusion.module.blocks import TimestepEmbedding, Timesteps
from cosmos1.models.diffusion.networks.general_dit import GeneralDIT
from cosmos1.utils import log


class VideoExtendGeneralDIT(GeneralDIT):
    def __init__(self, *args, in_channels=16 + 1, add_augment_sigma_embedding=False, **kwargs):
        self.add_augment_sigma_embedding = add_augment_sigma_embedding

        # extra channel for video condition mask
        super().__init__(*args, in_channels=in_channels, **kwargs)
        log.debug(f"VideoExtendGeneralDIT in_channels: {in_channels}")

    def build_additional_timestamp_embedder(self):
        super().build_additional_timestamp_embedder()
        if self.add_augment_sigma_embedding:
            log.info("Adding augment sigma embedding")
            self.augment_sigma_embedder = nn.Sequential(
                Timesteps(self.model_channels),
                TimestepEmbedding(self.model_channels, self.model_channels, use_adaln_lora=self.use_adaln_lora),
            )

    def initialize_weights(self):
        if self.add_augment_sigma_embedding:
            # Initialize timestep embedding for augment sigma
            nn.init.normal_(self.augment_sigma_embedder[1].linear_1.weight, std=0.02)
            if self.augment_sigma_embedder[1].linear_1.bias is not None:
                nn.init.constant_(self.augment_sigma_embedder[1].linear_1.bias, 0)
            nn.init.normal_(self.augment_sigma_embedder[1].linear_2.weight, std=0.02)
            if self.augment_sigma_embedder[1].linear_2.bias is not None:
                nn.init.constant_(self.augment_sigma_embedder[1].linear_2.bias, 0)

        super().initialize_weights()  # Call this last since it wil call TP weight init

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        video_cond_bool: Optional[torch.Tensor] = None,
        condition_video_indicator: Optional[torch.Tensor] = None,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        condition_video_augment_sigma: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass of the video-conditioned DIT model.

        Args:
            x: Input tensor of shape (B, C, T, H, W)
            timesteps: Timestep tensor of shape (B,)
            crossattn_emb: Cross attention embeddings of shape (B, N, D)
            crossattn_mask: Optional cross attention mask of shape (B, N)
            fps: Optional frames per second tensor
            image_size: Optional image size tensor
            padding_mask: Optional padding mask tensor
            scalar_feature: Optional scalar features tensor
            data_type: Type of data being processed (default: DataType.VIDEO)
            video_cond_bool: Optional video conditioning boolean tensor
            condition_video_indicator: Optional video condition indicator tensor
            condition_video_input_mask: Required mask tensor for video data type
            condition_video_augment_sigma: Optional sigma values for conditional input augmentation
            **kwargs: Additional keyword arguments

        Returns:
            torch.Tensor: Output tensor
        """
        B, C, T, H, W = x.shape

        if data_type == DataType.VIDEO:
            assert condition_video_input_mask is not None, "condition_video_input_mask is required for video data type"

            input_list = [x, condition_video_input_mask]
            x = torch.cat(
                input_list,
                dim=1,
            )

        return super().forward(
            x=x,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            crossattn_mask=crossattn_mask,
            fps=fps,
            image_size=image_size,
            padding_mask=padding_mask,
            scalar_feature=scalar_feature,
            data_type=data_type,
            condition_video_augment_sigma=condition_video_augment_sigma,
            **kwargs,
        )

    def forward_before_blocks(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        condition_video_augment_sigma: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks

            condition_video_augment_sigma: (B, T) tensor of sigma value for the conditional input augmentation
        """
        del kwargs
        assert isinstance(
            data_type, DataType
        ), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        original_shape = x.shape
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x,
            fps=fps,
            padding_mask=padding_mask,
            latent_condition=latent_condition,
            latent_condition_sigma=latent_condition_sigma,
        )
        # logging affline scale information
        affline_scale_log_info = {}

        timesteps_B_D, adaln_lora_B_3D = self.t_embedder(timesteps.flatten())
        affline_emb_B_D = timesteps_B_D
        affline_scale_log_info["timesteps_B_D"] = timesteps_B_D.detach()

        if scalar_feature is not None:
            raise NotImplementedError("Scalar feature is not implemented yet.")

        if self.add_augment_sigma_embedding:
            if condition_video_augment_sigma is None:
                # Handling image case
                # Note: for video case, when there is not condition frames, we also set it as zero, see extend_model augment_conditional_latent_frames function
                assert data_type == DataType.IMAGE, "condition_video_augment_sigma is required for video data type"
                condition_video_augment_sigma = torch.zeros_like(timesteps.flatten())

            affline_augment_sigma_emb_B_D, _ = self.augment_sigma_embedder(condition_video_augment_sigma.flatten())
            affline_emb_B_D = affline_emb_B_D + affline_augment_sigma_emb_B_D
        affline_scale_log_info["affline_emb_B_D"] = affline_emb_B_D.detach()
        affline_emb_B_D = self.affline_norm(affline_emb_B_D)

        if self.use_cross_attn_mask:
            crossattn_mask = crossattn_mask[:, None, None, :].to(dtype=torch.bool)  # [B, 1, 1, length]
        else:
            crossattn_mask = None

        x = rearrange(x_B_T_H_W_D, "B T H W D -> T H W B D")
        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = rearrange(
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, "B T H W D -> T H W B D"
            )
        crossattn_emb = rearrange(crossattn_emb, "B M D -> M B D")
        if crossattn_mask:
            crossattn_mask = rearrange(crossattn_mask, "B M -> M B")

        output = {
            "x": x,
            "affline_emb_B_D": affline_emb_B_D,
            "crossattn_emb": crossattn_emb,
            "crossattn_mask": crossattn_mask,
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_3D": adaln_lora_B_3D,
            "original_shape": original_shape,
            "extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        }
        return output
