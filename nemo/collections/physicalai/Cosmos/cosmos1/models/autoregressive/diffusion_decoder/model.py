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

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from cosmos1.models.diffusion.conditioner import BaseVideoCondition
from cosmos1.models.diffusion.diffusion.functional.batch_ops import batch_mul
from cosmos1.models.diffusion.diffusion.modules.res_sampler import COMMON_SOLVER_OPTIONS
from cosmos1.models.diffusion.model.model_t2w import DiffusionT2WModel as VideoDiffusionModel
from cosmos1.utils.lazy_config import instantiate as lazy_instantiate


@dataclass
class VideoLatentDiffusionDecoderCondition(BaseVideoCondition):
    # latent_condition will concat to the input of network, along channel dim;
    # cfg will make latent_condition all zero padding.
    latent_condition: Optional[torch.Tensor] = None
    latent_condition_sigma: Optional[torch.Tensor] = None


class LatentDiffusionDecoderModel(VideoDiffusionModel):
    def __init__(self, config):
        super().__init__(config)
        """
        latent_corruptor: the corruption module is used to corrupt the latents. It add gaussian noise to the latents.
        pixel_corruptor: the corruption module is used to corrupt the pixels. It apply gaussian blur kernel to pixels in a temporal consistent way.
        tokenizer_corruptor: the corruption module is used to simulate tokenizer reconstruction errors.

        diffusion decoder noise augmentation pipeline for continuous token condition model:
        condition: GT_video [T, H, W]
                        -> tokenizer_corruptor~(8x8x8) encode -> latent_corruptor -> tokenizer_corruptor~(8x8x8) decode
                        -> pixel corruptor
                        -> tokenizer~(1x8x8) encode -> condition [T, H/8, W/8]
        GT: GT_video [T, H, W] -> tokenizer~(1x8x8) -> x_t [T, H/8, W/8].

        diffusion decoder noise augmentation pipeline for discrete token condition model:
        condition: GT_video [T, H, W]
                -> pixel corruptor
                -> discrete tokenizer encode -> condition [T, T/8, H/16, W/16]
        GT: GT_video [T, H, W] -> tokenizer~(8x8x8) -> x_t [T, T/8, H/8, W/8].

        """
        self.latent_corruptor = lazy_instantiate(config.latent_corruptor)
        self.pixel_corruptor = lazy_instantiate(config.pixel_corruptor)
        self.tokenizer_corruptor = lazy_instantiate(config.tokenizer_corruptor)

        if self.latent_corruptor:
            self.latent_corruptor.to(**self.tensor_kwargs)
        if self.pixel_corruptor:
            self.pixel_corruptor.to(**self.tensor_kwargs)

        if self.tokenizer_corruptor:
            if hasattr(self.tokenizer_corruptor, "reset_dtype"):
                self.tokenizer_corruptor.reset_dtype()
        else:
            assert self.pixel_corruptor is not None

        self.diffusion_decoder_cond_sigma_low = config.diffusion_decoder_cond_sigma_low
        self.diffusion_decoder_cond_sigma_high = config.diffusion_decoder_cond_sigma_high
        self.diffusion_decoder_corrupt_prob = config.diffusion_decoder_corrupt_prob
        if hasattr(config, "condition_on_tokenizer_corruptor_token"):
            self.condition_on_tokenizer_corruptor_token = config.condition_on_tokenizer_corruptor_token
        else:
            self.condition_on_tokenizer_corruptor_token = False

    def is_image_batch(self, data_batch: dict[str, Tensor]) -> bool:
        """We hanlde two types of data_batch. One comes from a joint_dataloader where "dataset_name" can be used to differenciate image_batch and video_batch.
        Another comes from a dataloader which we by default assumes as video_data for video model training.
        """
        is_image = self.input_image_key in data_batch
        is_video = self.input_data_key in data_batch
        assert (
            is_image != is_video
        ), "Only one of the input_image_key or input_data_key should be present in the data_batch."
        return is_image

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        apply_corruptor: bool = True,
        corrupt_sigma: float = 1.5,
        preencode_condition: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        input_key = self.input_data_key  # by default it is video key
        # Latent state
        raw_state = data_batch[input_key]

        if self.condition_on_tokenizer_corruptor_token:
            if preencode_condition:
                latent_condition = raw_state.to(torch.int32).contiguous()
                corrupted_pixel = self.tokenizer_corruptor.decode(latent_condition[:, 0])
            else:
                corrupted_pixel = (
                    self.pixel_corruptor(raw_state) if apply_corruptor and self.pixel_corruptor else raw_state
                )
                latent_condition = self.tokenizer_corruptor.encode(corrupted_pixel)
                latent_condition = latent_condition[1] if isinstance(latent_condition, tuple) else latent_condition
                corrupted_pixel = self.tokenizer_corruptor.decode(latent_condition)
                latent_condition = latent_condition.unsqueeze(1)
        else:
            if preencode_condition:
                latent_condition = raw_state
                corrupted_pixel = self.decode(latent_condition)
            else:
                corrupted_pixel = (
                    self.pixel_corruptor(raw_state) if apply_corruptor and self.pixel_corruptor else raw_state
                )
                latent_condition = self.encode(corrupted_pixel).contiguous()

        sigma = (
            torch.rand((latent_condition.shape[0],)).to(**self.tensor_kwargs) * corrupt_sigma
        )  # small value to indicate clean video
        _, _, _, c_noise_cond = self.scaling(sigma=sigma)
        if corrupt_sigma != self.diffusion_decoder_cond_sigma_low and self.diffusion_decoder_corrupt_prob > 0:
            noise = batch_mul(sigma, torch.randn_like(latent_condition))
            latent_condition = latent_condition + noise
        data_batch["latent_condition_sigma"] = batch_mul(torch.ones_like(latent_condition[:, 0:1, ::]), c_noise_cond)
        data_batch["latent_condition"] = latent_condition
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(noise_x, sigma, condition).x0
            uncond_x0 = self.denoise(noise_x, sigma, uncondition).x0
            return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return x0_fn, corrupted_pixel

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        solver_option: COMMON_SOLVER_OPTIONS = "2ab",
        sigma_min: float = 0.02,
        apply_corruptor: bool = False,
        return_recon_x: bool = False,
        corrupt_sigma: float = 0.01,
        preencode_condition: bool = False,
    ) -> Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to self.state_shape if not provided
            n_sample (int): number of samples to generate
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            num_steps (int): number of steps for the diffusion process
            solver_option (str): differential equation solver option, default to "2ab"~(mulitstep solver)
            preencode_condition (bool): use pre-computed condition if true, save tokenizer's inference time memory/
        """
        if not preencode_condition:
            self._normalize_video_databatch_inplace(data_batch)
            self._augment_image_dim_inplace(data_batch)
        is_image_batch = False
        if n_sample is None:
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            if is_image_batch:
                state_shape = (self.state_shape[0], 1, *self.state_shape[2:])  # C,T,H,W

        x0_fn, recon_x = self.get_x0_fn_from_batch(
            data_batch,
            guidance,
            is_negative_prompt=is_negative_prompt,
            apply_corruptor=apply_corruptor,
            corrupt_sigma=corrupt_sigma,
            preencode_condition=preencode_condition,
        )
        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)
        x_sigma_max = (
            torch.randn(n_sample, *state_shape, **self.tensor_kwargs, generator=generator) * self.sde.sigma_max
        )

        samples = self.sampler(
            x0_fn,
            x_sigma_max,
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=self.sde.sigma_max,
            solver_option=solver_option,
        )

        if return_recon_x:
            return samples, recon_x
        else:
            return samples
