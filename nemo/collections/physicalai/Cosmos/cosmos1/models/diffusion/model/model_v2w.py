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

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from cosmos1.models.diffusion.conditioner import VideoExtendCondition
from cosmos1.models.diffusion.config.base.conditioner import VideoCondBoolConfig
from cosmos1.models.diffusion.diffusion.functional.batch_ops import batch_mul
from cosmos1.models.diffusion.model.model_t2w import DiffusionT2WModel
from cosmos1.utils import log, misc
from torch import Tensor


@dataclass
class VideoDenoisePrediction:
    x0: torch.Tensor  # clean data prediction
    eps: Optional[torch.Tensor] = None  # noise prediction
    logvar: Optional[torch.Tensor] = None  # log variance of noise prediction, can be used a confidence / uncertainty
    xt: Optional[torch.Tensor] = None  # input to the network, before muliply with c_in
    x0_pred_replaced: Optional[torch.Tensor] = None  # x0 prediction with condition region replaced by gt_latent


class DiffusionV2WModel(DiffusionT2WModel):
    def __init__(self, config):
        super().__init__(config)

    def augment_conditional_latent_frames(
        self,
        condition: VideoExtendCondition,
        cfg_video_cond_bool: VideoCondBoolConfig,
        gt_latent: Tensor,
        condition_video_augment_sigma_in_inference: float = 0.001,
        sigma: Tensor = None,
        seed: int = 1,
    ) -> Union[VideoExtendCondition, Tensor]:
        """Augments the conditional frames with noise during inference.

        Args:
            condition (VideoExtendCondition): condition object
                condition_video_indicator: binary tensor indicating the region is condition(value=1) or generation(value=0). Bx1xTx1x1 tensor.
                condition_video_input_mask: input mask for the network input, indicating the condition region. B,1,T,H,W tensor. will be concat with the input for the network.
            cfg_video_cond_bool (VideoCondBoolConfig): video condition bool config
            gt_latent (Tensor): ground truth latent tensor in shape B,C,T,H,W
            condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
            sigma (Tensor): noise level for the generation region
            seed (int): random seed for reproducibility
        Returns:
            VideoExtendCondition: updated condition object
                condition_video_augment_sigma: sigma for the condition region, feed to the network
            augment_latent (Tensor): augmented latent tensor in shape B,C,T,H,W

        """

        # Inference only, use fixed sigma for the condition region
        assert (
            condition_video_augment_sigma_in_inference is not None
        ), "condition_video_augment_sigma_in_inference should be provided"
        augment_sigma = condition_video_augment_sigma_in_inference

        if augment_sigma >= sigma.flatten()[0]:
            # This is a inference trick! If the sampling sigma is smaller than the augment sigma, we will start denoising the condition region together.
            # This is achieved by setting all region as `generation`, i.e. value=0
            log.debug("augment_sigma larger than sigma or other frame, remove condition")
            condition.condition_video_indicator = condition.condition_video_indicator * 0

        augment_sigma = torch.tensor([augment_sigma], **self.tensor_kwargs)

        # Now apply the augment_sigma to the gt_latent

        noise = misc.arch_invariant_rand(
            gt_latent.shape,
            torch.float32,
            self.tensor_kwargs["device"],
            seed,
        )

        augment_latent = gt_latent + noise * augment_sigma[:, None, None, None, None]

        _, _, c_in_augment, _ = self.scaling(sigma=augment_sigma)

        # Multiply the whole latent with c_in_augment
        augment_latent_cin = batch_mul(augment_latent, c_in_augment)

        # Since the whole latent will multiply with c_in later, we devide the value to cancel the effect
        _, _, c_in, _ = self.scaling(sigma=sigma)
        augment_latent_cin = batch_mul(augment_latent_cin, 1 / c_in)

        return condition, augment_latent_cin

    def denoise(
        self,
        noise_x: Tensor,
        sigma: Tensor,
        condition: VideoExtendCondition,
        condition_video_augment_sigma_in_inference: float = 0.001,
        seed: int = 1,
    ) -> VideoDenoisePrediction:
        """Denoises input tensor using conditional video generation.

        Args:
            noise_x (Tensor): Noisy input tensor.
            sigma (Tensor): Noise level.
            condition (VideoExtendCondition): Condition for denoising.
            condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
            seed (int): Random seed for reproducibility
        Returns:
            VideoDenoisePrediction containing:
            - x0: Denoised prediction
            - eps: Noise prediction
            - logvar: Log variance of noise prediction
            - xt: Input before c_in multiplication
            - x0_pred_replaced: x0 prediction with condition regions replaced by ground truth
        """

        assert (
            condition.gt_latent is not None
        ), f"find None gt_latent in condition, likely didn't call self.add_condition_video_indicator_and_video_input_mask when preparing the condition or this is a image batch but condition.data_type is wrong, get {noise_x.shape}"
        gt_latent = condition.gt_latent
        cfg_video_cond_bool: VideoCondBoolConfig = self.config.conditioner.video_cond_bool

        condition_latent = gt_latent

        # Augment the latent with different sigma value, and add the augment_sigma to the condition object if needed
        condition, augment_latent = self.augment_conditional_latent_frames(
            condition, cfg_video_cond_bool, condition_latent, condition_video_augment_sigma_in_inference, sigma, seed
        )
        condition_video_indicator = condition.condition_video_indicator  # [B, 1, T, 1, 1]
        # Compose the model input with condition region (augment_latent) and generation region (noise_x)
        new_noise_xt = condition_video_indicator * augment_latent + (1 - condition_video_indicator) * noise_x
        # Call the base model
        denoise_pred = super().denoise(new_noise_xt, sigma, condition)

        x0_pred_replaced = condition_video_indicator * gt_latent + (1 - condition_video_indicator) * denoise_pred.x0

        x0_pred = x0_pred_replaced

        return VideoDenoisePrediction(
            x0=x0_pred,
            eps=batch_mul(noise_x - x0_pred, 1.0 / sigma),
            logvar=denoise_pred.logvar,
            xt=new_noise_xt,
            x0_pred_replaced=x0_pred_replaced,
        )

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        condition_latent: Union[torch.Tensor, None] = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
        x_sigma_max: Optional[torch.Tensor] = None,
    ) -> Tensor:
        """Generates video samples conditioned on input frames.

        Args:
            data_batch: Input data dictionary
            guidance: Classifier-free guidance scale
            seed: Random seed for reproducibility
            state_shape: Shape of output tensor (defaults to model's state shape)
            n_sample: Number of samples to generate (defaults to batch size)
            is_negative_prompt: Whether to use negative prompting
            num_steps: Number of denoising steps
            condition_latent: Conditioning frames tensor (B,C,T,H,W)
            num_condition_t: Number of frames to condition on
            condition_video_augment_sigma_in_inference: Noise level for condition augmentation
            add_input_frames_guidance: Whether to apply guidance to input frames
            x_sigma_max: Maximum noise level tensor

        Returns:
            Generated video samples tensor
        """

        if n_sample is None:
            input_key = self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            log.debug(f"Default Video state shape is used. {self.state_shape}")
            state_shape = self.state_shape

        assert condition_latent is not None, "condition_latent should be provided"

        x0_fn = self.get_x0_fn_from_batch_with_condition_latent(
            data_batch,
            guidance,
            is_negative_prompt=is_negative_prompt,
            condition_latent=condition_latent,
            num_condition_t=num_condition_t,
            condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            add_input_frames_guidance=add_input_frames_guidance,
            seed=seed,
        )
        if x_sigma_max is None:
            x_sigma_max = (
                misc.arch_invariant_rand(
                    (n_sample,) + tuple(state_shape),
                    torch.float32,
                    self.tensor_kwargs["device"],
                    seed,
                )
                * self.sde.sigma_max
            )

        samples = self.sampler(x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=self.sde.sigma_max)
        return samples

    def get_x0_fn_from_batch_with_condition_latent(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        condition_latent: torch.Tensor = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
        seed: int = 1,
    ) -> Callable:
        """Creates denoising function for conditional video generation.

        Args:
            data_batch: Input data dictionary
            guidance: Classifier-free guidance scale
            is_negative_prompt: Whether to use negative prompting
            condition_latent: Conditioning frames tensor (B,C,T,H,W)
            num_condition_t: Number of frames to condition on
            condition_video_augment_sigma_in_inference: Noise level for condition augmentation
            add_input_frames_guidance: Whether to apply guidance to input frames
            seed: Random seed for reproducibility

        Returns:
            Function that takes noisy input and noise level and returns denoised prediction
        """
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        condition.video_cond_bool = True
        condition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, condition, num_condition_t
        )

        uncondition.video_cond_bool = False if add_input_frames_guidance else True
        uncondition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, uncondition, num_condition_t
        )

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(
                noise_x,
                sigma,
                condition,
                condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                seed=seed,
            ).x0_pred_replaced
            uncond_x0 = self.denoise(
                noise_x,
                sigma,
                uncondition,
                condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                seed=seed,
            ).x0_pred_replaced

            return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return x0_fn

    def add_condition_video_indicator_and_video_input_mask(
        self, latent_state: torch.Tensor, condition: VideoExtendCondition, num_condition_t: Union[int, None] = None
    ) -> VideoExtendCondition:
        """Adds conditioning masks to VideoExtendCondition object.

        Creates binary indicators and input masks for conditional video generation.

        Args:
            latent_state: Input latent tensor (B,C,T,H,W)
            condition: VideoExtendCondition object to update
            num_condition_t: Number of frames to condition on

        Returns:
            Updated VideoExtendCondition with added masks:
            - condition_video_indicator: Binary tensor marking condition regions
            - condition_video_input_mask: Input mask for network
            - gt_latent: Ground truth latent tensor
        """
        T = latent_state.shape[2]
        latent_dtype = latent_state.dtype
        condition_video_indicator = torch.zeros(1, 1, T, 1, 1, device=latent_state.device).type(
            latent_dtype
        )  # 1 for condition region

        # Only in inference to decide the condition region
        assert num_condition_t is not None, "num_condition_t should be provided"
        assert num_condition_t <= T, f"num_condition_t should be less than T, get {num_condition_t}, {T}"
        log.debug(
            f"condition_location first_n, num_condition_t {num_condition_t}, condition.video_cond_bool {condition.video_cond_bool}"
        )
        condition_video_indicator[:, :, :num_condition_t] += 1.0

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
