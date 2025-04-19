# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# pylint: disable=C0115,C0116,C0301

from typing import Union

import torch
from cosmos1.models.diffusion.conditioner import VideoExtendCondition
from cosmos1.models.diffusion.model.model_t2w import DiffusionT2WModel
from cosmos1.models.diffusion.model.model_v2w import DiffusionV2WModel
from cosmos1.utils import log
from einops import rearrange


class DiffusionMultiCameraT2WModel(DiffusionT2WModel):
    def __init__(self, config):
        super().__init__(config)
        self.n_cameras = config.net.n_cameras

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        state = rearrange(state, "B C (V T) H W -> (B V) C T H W", V=self.n_cameras)
        encoded_state = self.tokenizer.encode(state)
        encoded_state = rearrange(encoded_state, "(B V) C T H W -> B C (V T) H W", V=self.n_cameras) * self.sigma_data
        return encoded_state

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=self.n_cameras)
        decoded_state = self.tokenizer.decode(latent / self.sigma_data)
        decoded_state = rearrange(decoded_state, "(B V) C T H W -> B C (V T) H W", V=self.n_cameras)
        return decoded_state


class DiffusionMultiCameraV2WModel(DiffusionV2WModel):
    def __init__(self, config):
        super().__init__(config)
        self.n_cameras = config.net.n_cameras

    #
    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        state = rearrange(state, "B C (V T) H W -> (B V) C T H W", V=self.n_cameras)
        encoded_state = self.tokenizer.encode(state)
        encoded_state = rearrange(encoded_state, "(B V) C T H W -> B C (V T) H W", V=self.n_cameras) * self.sigma_data
        return encoded_state

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=self.n_cameras)
        decoded_state = self.tokenizer.decode(latent / self.sigma_data)
        decoded_state = rearrange(decoded_state, "(B V) C T H W -> B C (V T) H W", V=self.n_cameras)
        return decoded_state

    def add_condition_video_indicator_and_video_input_mask(
        self, latent_state: torch.Tensor, condition: VideoExtendCondition, num_condition_t: Union[int, None] = None
    ) -> VideoExtendCondition:
        """Add condition_video_indicator and condition_video_input_mask to the condition object for video conditioning.
        condition_video_indicator is a binary tensor indicating the condition region in the latent state. 1x1xTx1x1 tensor.
        condition_video_input_mask will be concat with the input for the network.
        Args:
            latent_state (torch.Tensor): latent state tensor in shape B,C,T,H,W
            condition (VideoExtendCondition): condition object
            num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        Returns:
            VideoExtendCondition: updated condition object
        """
        T = latent_state.shape[2]
        latent_dtype = latent_state.dtype
        condition_video_indicator = torch.zeros(1, 1, T, 1, 1, device=latent_state.device).type(
            latent_dtype
        )  # 1 for condition region

        condition_video_indicator = rearrange(
            condition_video_indicator, "B C (V T) H W -> (B V) C T H W", V=self.n_cameras
        )
        # Only in inference to decide the condition region
        assert num_condition_t is not None, "num_condition_t should be provided"
        assert num_condition_t <= T, f"num_condition_t should be less than T, get {num_condition_t}, {T}"
        log.info(
            f"condition_location first_n, num_condition_t {num_condition_t}, condition.video_cond_bool {condition.video_cond_bool}"
        )
        condition_video_indicator[:, :, :num_condition_t] += 1.0
        condition_video_indicator = rearrange(
            condition_video_indicator, "(B V) C T H W -> B C (V T) H W", V=self.n_cameras
        )

        condition.gt_latent = latent_state
        condition.condition_video_indicator = condition_video_indicator

        B, C, T, H, W = latent_state.shape
        # Create additional input_mask channel, this will be concatenated to the input of the network
        ones_padding = torch.ones((B, 1, T, H, W), dtype=latent_state.dtype, device=latent_state.device)
        zeros_padding = torch.zeros((B, 1, T, H, W), dtype=latent_state.dtype, device=latent_state.device)
        assert condition.video_cond_bool is not None, "video_cond_bool should be set"

        # The input mask indicate whether the input is conditional region or not
        if condition.video_cond_bool:  # Condition one given video frames
            condition.condition_video_input_mask = (
                condition_video_indicator * ones_padding + (1 - condition_video_indicator) * zeros_padding
            )
        else:  # Unconditional case, use for cfg
            condition.condition_video_input_mask = zeros_padding

        return condition
