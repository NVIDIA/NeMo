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

# pylint: disable=C0115,C0301


from typing import Callable, Dict, Tuple, Union

import numpy as np
import torch
import torch.distributed
from megatron.core import parallel_state
from torch import Tensor

from nemo.collections.diffusion.sampler.context_parallel import cat_outputs_cp, split_inputs_cp
from nemo.collections.diffusion.sampler.cosmos.cosmos_extended_diffusion_pipeline import ExtendedDiffusionPipeline
from nemo.collections.diffusion.sampler.res.res_sampler import COMMON_SOLVER_OPTIONS


class CosmosControlDiffusionPipeline(ExtendedDiffusionPipeline):
    def __init__(self, *args, **kwargs):
        base_model = kwargs.pop("base_model", None)
        super().__init__(*args, **kwargs)
        self.base_model = base_model
        self.net.sigma_data = self.sigma_data

    def get_x0_fn_from_batch_with_condition_latent(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        condition_latent: torch.Tensor = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            condition_latent (torch.Tensor): latent tensor in shape B,C,T,H,W as condition to generate video.
        - num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        - condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        # data_batch should be the one processed by self.get_data_and_condition
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
            # Add conditions for long video generation.

        if condition_latent is None:
            condition_latent = torch.zeros(data_batch["latent_hint"].shape, **self.tensor_kwargs)
            num_condition_t = 0
            condition_video_augment_sigma_in_inference = 1000

        condition.video_cond_bool = True
        condition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, condition, num_condition_t
        )

        uncondition.video_cond_bool = True  # Not do cfg on condition frames
        uncondition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, uncondition, num_condition_t
        )

        # Add extra conditions for ctrlnet.
        latent_hint = data_batch["latent_hint"]
        hint_key = data_batch["hint_key"]
        setattr(condition, hint_key, latent_hint)
        setattr(uncondition, "hint_key", hint_key)
        setattr(condition, "hint_key", hint_key)
        setattr(condition, "base_model", self.base_model)
        setattr(uncondition, "base_model", self.base_model)

        if "use_none_hint" in data_batch and data_batch["use_none_hint"]:
            setattr(uncondition, hint_key, None)
        else:
            setattr(uncondition, hint_key, latent_hint)

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            condition.base_model = self.base_model
            uncondition.base_model = self.base_model
            cond_x0, eps_pred_cond, logvar_cond = self.denoise(
                noise_x,
                sigma,
                condition,
                condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            )
            uncond_x0, eps_pred_uncond, logvar_uncond = self.denoise(
                noise_x,
                sigma,
                uncondition,
                condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            )
            return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return x0_fn

    def get_x_from_clean(
        self,
        in_clean_img: torch.Tensor,
        sigma_max: float | None,
        seed: int = 1,
    ) -> Tensor:
        """
        in_clean_img (torch.Tensor): input clean image for image-to-image/video-to-video by adding noise then denoising
        sigma_max (float): maximum sigma applied to in_clean_image for image-to-image/video-to-video
        """
        if in_clean_img is None:
            return None
        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)
        noise = torch.randn(*in_clean_img.shape, **self.tensor_kwargs, generator=generator)
        if sigma_max is None:
            sigma_max = self.sde.sigma_max
        x_sigma_max = in_clean_img + noise * sigma_max
        return x_sigma_max

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        condition_latent: Union[torch.tensor, None] = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
        solver_option: COMMON_SOLVER_OPTIONS = "2ab",
    ) -> Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        """

        is_image_batch = self.is_image_batch(data_batch)
        if n_sample is None:
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            n_sample = data_batch[input_key].shape[0]

        if self._noise_generator is None:
            self._initialize_generators()

        state_shape = list(state_shape)
        np.random.seed(self.seed)
        x_sigma_max = self.get_x_from_clean((condition_latent), self.sde.sigma_max)

        cp_enabled = parallel_state.get_context_parallel_world_size() > 1
        condition_latent = torch.zeros_like(x_sigma_max)
        data_batch['condition_latent'] = condition_latent
        x0_fn = self.get_x0_fn_from_batch_with_condition_latent(
            data_batch,
            guidance,
            is_negative_prompt=is_negative_prompt,
            condition_latent=condition_latent,
            num_condition_t=num_condition_t,
            condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            # add_input_frames_guidance=add_input_frames_guidance,
        )

        if cp_enabled:
            cp_group = parallel_state.get_context_parallel_group()
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=cp_group)

        samples = None
        if self.sampler_type == "EDM":
            samples = self.sampler(x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=self.sde.sigma_max)
        elif self.sampler_type == "RES":
            samples = self.sampler(
                x0_fn, x_sigma_max, sigma_max=self.sde.sigma_max, num_steps=num_steps, solver_option=solver_option
            )

        if cp_enabled:
            cp_group = parallel_state.get_context_parallel_group()
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=cp_group)

        return samples
