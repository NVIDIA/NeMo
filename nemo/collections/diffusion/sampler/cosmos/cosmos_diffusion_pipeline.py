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

# pylint: disable=C0116,C0301

import warnings
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.distributed
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from nemo.collections.diffusion.sampler.batch_ops import *
from nemo.collections.diffusion.sampler.conditioner import BaseVideoCondition, DataType, Edify4Condition
from nemo.collections.diffusion.sampler.context_parallel import cat_outputs_cp, split_inputs_cp
from nemo.collections.diffusion.sampler.edm.edm import EDMSDE, EDMSampler, EDMScaling
from nemo.collections.diffusion.sampler.edm.edm_pipeline import EDMPipeline
from nemo.collections.diffusion.sampler.res.res_sampler import COMMON_SOLVER_OPTIONS, RESSampler

# key to check if the video data is normalized or image data is converted to video data
# to avoid apply normalization or augment image dimension multiple times
# It is due to we do not have normalization and augment image dimension in the dataloader and move it to the model
IS_PREPROCESSED_KEY = "is_preprocessed"


class CosmosDiffusionPipeline(EDMPipeline):
    """
    Diffusion pipeline for Cosmos Diffusion model.
    """

    def __init__(
        self,
        # Video Tokenizer
        # DiT Model
        net=None,
        # Conditioning Embedders
        conditioner=None,
        vae=None,
        # SDE Args
        p_mean=0.0,
        p_std=1.0,
        sigma_max=80,
        sigma_min=0.0002,
        sampler_type="RES",  # or "EDM"
        # EDM Scaling Args
        sigma_data=0.5,
        seed=1,
        loss_add_logvar=True,
    ):

        super().__init__(
            net,
        )
        self.vae = vae
        self.net = net
        self.conditioner = conditioner

        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.sampler_type = sampler_type

        self.seed = seed
        self._noise_generator = None
        self._noise_level_generator = None

        self.sde = EDMSDE(p_mean, p_std, sigma_max, sigma_min)

        if self.sampler_type == "EDM":
            self.sampler = EDMSampler()
        elif self.sampler_type == "RES":
            self.sampler = RESSampler()

        self.scaling = EDMScaling(sigma_data)

        self.input_data_key = 'video'
        self.input_image_key = 'images_1024'
        self.tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}
        self.loss_reduce = 'mean'

        self.loss_add_logvar = loss_add_logvar
        self.loss_scale = 1.0

        self.aesthetic_finetuning = None
        self.camera_sample_weight = None
        self.loss_mask_enabled = False

    @property
    def noise_level_generator(self):
        return self._noise_level_generator

    def _initialize_generators(self):
        noise_seed = self.seed + 100 * parallel_state.get_data_parallel_rank(with_context_parallel=True)
        noise_level_seed = self.seed + 100 * parallel_state.get_data_parallel_rank(with_context_parallel=False)
        self._noise_generator = torch.Generator(device='cuda')
        self._noise_generator.manual_seed(noise_seed)
        self._noise_level_generator = np.random.default_rng(noise_level_seed)
        self.sde._generator = self._noise_level_generator

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

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(state) * self.sigma_data

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent / self.sigma_data)

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the diffusion model.

        This method is responsible for executing one iteration of the model's training. It involves:
        1. Adding noise to the input data using the SDE process.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data, \
            considering any configured loss weighting.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.

        Returns:
            tuple: A tuple containing two elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor: The computed loss for the training step as a PyTorch Tensor.

        Raises:
            AssertionError: If the class is conditional, \
                but no number of classes is specified in the network configuration.

        Notes:
            - The method handles different types of conditioning
            - The method also supports Kendall's loss
        """
        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        x0_from_data_batch, x0, condition = self.get_data_and_condition(data_batch)

        # Sample pertubation noise levels and N(0, 1) noises
        sigma, epsilon = self.draw_training_sigma_and_epsilon(x0.size(), condition)

        if parallel_state.is_pipeline_last_stage():
            output_batch, kendall_loss, pred_mse, edm_loss = self.compute_loss_with_epsilon_and_sigma(
                data_batch, x0_from_data_batch, x0, condition, epsilon, sigma
            )
            return output_batch, kendall_loss
        else:
            net_output = self.compute_loss_with_epsilon_and_sigma(
                data_batch, x0_from_data_batch, x0, condition, epsilon, sigma
            )
            return net_output

    def denoise(self, xt: torch.Tensor, sigma: torch.Tensor, condition: Edify4Condition):
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (Edify4Condition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred) and optional confidence (logvar).
        """

        # Currently only supports video.
        # if self.config.get("use_dummy_temporal_dim", False):
        #     # When using video DiT model for image, we need to use a dummy temporal dimension.
        #     xt = xt.unsqueeze(2)

        xt = xt.to(**self.tensor_kwargs)
        sigma = sigma.to(**self.tensor_kwargs)
        # get precondition for the network
        c_skip, c_out, c_in, c_noise = self.scaling(sigma=sigma)

        output = self.net(
            x=batch_mul(c_in, xt),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps=c_noise,  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        )
        if isinstance(output, tuple):
            net_output, logvar = output
        else:
            net_output, logvar = output, None
        if not parallel_state.is_pipeline_last_stage():
            return net_output

        # logvar = self.net.compute_logvar(c_noise)

        x0_pred = batch_mul(c_skip, xt) + batch_mul(c_out, net_output)

        # get noise prediction based on sde
        eps_pred = batch_mul(xt - x0_pred, 1.0 / sigma)

        # Currently only supports video.
        # if self.config.get("use_dummy_temporal_dim", False):
        #     x0_pred = x0_pred.squeeze(2)
        #     eps_pred = eps_pred.squeeze(2)

        return x0_pred, eps_pred, logvar

    def compute_loss_with_epsilon_and_sigma(
        self,
        data_batch: dict[str, torch.Tensor],
        x0_from_data_batch: torch.Tensor,
        x0: torch.Tensor,
        condition: Edify4Condition,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
    ):
        """
        Compute loss givee epsilon and sigma

        This method is responsible for computing loss give epsilon and sigma. It involves:
        1. Adding noise to the input data using the SDE process.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data, \
            considering any configured loss weighting.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            x0_from_data_batch: raw image/video
            x0: image/video latent
            condition: text condition
            epsilon: noise
            sigma: noise level

        Returns:
            tuple: A tuple containing four elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor 1: kendall loss,
                - Tensor 2: MSE loss,
                - Tensor 3: EDM loss

        Raises:
            AssertionError: If the class is conditional, \
                but no number of classes is specified in the network configuration.

        Notes:
            - The method handles different types of conditioning
            - The method also supports Kendall's loss
        """
        # Get the mean and stand deviation of the marginal probability distribution.
        mean, std = self.sde.marginal_prob(x0, sigma)
        # Generate noisy observations
        xt = mean + batch_mul(std, epsilon)  # corrupted data

        if parallel_state.is_pipeline_last_stage():
            # make prediction
            x0_pred, eps_pred, logvar = self.denoise(xt, sigma, condition)
            # loss weights for different noise levels
            weights_per_sigma = self.get_per_sigma_loss_weights(sigma=sigma)
            # extra weight for each sample, for example, aesthetic weight, camera weight
            weights_per_sample = self.get_per_sample_weight(data_batch, x0.shape[0])
            loss_mask_per_sample = 1.0
            pred_mse = (x0 - x0_pred) ** 2 * loss_mask_per_sample
            edm_loss = batch_mul(pred_mse, weights_per_sigma * weights_per_sample)
            if len(edm_loss.shape) > 5:
                edm_loss = edm_loss.squeeze(0)
            b, c, t, h, w = edm_loss.shape
            if logvar is not None and self.loss_add_logvar:
                kendall_loss = batch_mul(edm_loss, torch.exp(-logvar).view(-1)).flatten(start_dim=1) + logvar.view(
                    -1, 1
                )
            else:
                kendall_loss = edm_loss.flatten(start_dim=1)

            kendall_loss = rearrange(kendall_loss, "b (c t h w) -> b c (t h w)", b=b, c=c, t=t, h=h, w=w)

            output_batch = {
                "x0": x0,
                "xt": xt,
                "sigma": sigma,
                "weights_per_sigma": weights_per_sigma,
                "weights_per_sample": weights_per_sample,
                "loss_mask_per_sample": loss_mask_per_sample,
                "condition": condition,
                "model_pred": {"x0_pred": x0_pred, "eps_pred": eps_pred, "logvar": logvar},
                "mse_loss": pred_mse.mean(),
                "edm_loss": edm_loss.mean(),
            }
            return output_batch, kendall_loss, pred_mse, edm_loss
        else:
            # make prediction
            x0_pred = self.denoise(xt, sigma, condition)
            return x0_pred.contiguous()

    def get_per_sample_weight(self, data_batch: dict[str, torch.Tensor], batch_size: int):
        r"""
        extra weight for each sample, for example, aesthetic weight
        Args:
            data_batch: raw data batch draw from the training data loader.
            batch_size: int, the batch size of the input data
        """
        aesthetic_cfg = self.aesthetic_finetuning
        if (aesthetic_cfg is not None) and getattr(aesthetic_cfg, "enabled", False):
            sample_weight = data_batch["aesthetic_weight"]
        else:
            sample_weight = torch.ones(batch_size, **self.tensor_kwargs)

        camera_cfg = self.camera_sample_weight
        if (camera_cfg is not None) and getattr(camera_cfg, "enabled", False):
            sample_weight *= 1 + (data_batch["camera_attributes"][:, 1:].sum(dim=1) != 0) * (camera_cfg.weight - 1)
        return sample_weight

    def get_per_sigma_loss_weights(self, sigma: torch.Tensor):
        """
        Args:
            sigma (tensor): noise level

        Returns:
            loss weights per sigma noise level
        """
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
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
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0, _, _ = self.denoise(noise_x, sigma, condition)
            uncond_x0, _, _ = self.denoise(noise_x, sigma, uncondition)
            return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return x0_fn

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
    ) -> Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        """

        is_image_batch = self.is_image_batch(data_batch)
        if n_sample is None:
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            if is_image_batch:
                state_shape = (self.state_shape[0], 1, *self.state_shape[2:])  # C,T,H,W

        cp_enabled = parallel_state.get_context_parallel_world_size() > 1

        if self._noise_generator is None:
            self._initialize_generators()

        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)

        state_shape = list(state_shape)

        if len(state_shape) == 4:
            state_shape = [1] + state_shape
        np.random.seed(self.seed)
        x_sigma_max = (
            torch.from_numpy(np.random.randn(*state_shape).astype(np.float32)).to(
                dtype=torch.float32, device=self.tensor_kwargs["device"]
            )
            * self.sde.sigma_max
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

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor]) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Warning:
            A warning is issued if the data has not been previously normalized.

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the 'input_data_key' tensor within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        input_key = self.input_data_key
        # only handle video batch
        if input_key in data_batch:
            # Check if the data has already been normalized and avoid re-normalizing
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                return
            else:
                data_batch[input_key] = data_batch[input_key].to(dtype=torch.uint8)
                assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
                warnings.warn("Normalizing video data in-place.")
                data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

    def _augment_image_dim_inplace(self, data_batch: dict[str, Tensor]) -> None:
        input_key = self.input_image_key
        if input_key in data_batch:
            # Check if the data has already been augmented and avoid re-augmenting
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert (
                    data_batch[input_key].shape[2] == 1
                ), f"Image data is claimed be augmented while its shape is {data_batch[input_key].shape}"
                return
            else:
                data_batch[input_key] = rearrange(data_batch[input_key], "b c h w -> b c 1 h w").contiguous()
                data_batch[IS_PREPROCESSED_KEY] = True

    def get_data_and_condition(self, data_batch: dict[str, Tensor]) -> Tuple[Tensor, BaseVideoCondition]:
        """
        Retrieves data and conditioning for model input.

        Args:
            data_batch: Batch of input data.

        Returns:
            Raw data, latent data, and conditioning information.
        """
        # Extract video tensor
        raw_state = data_batch["video"] * self.sigma_data
        # Assume data is already encoded
        latent_state = raw_state

        # Condition
        condition = self.conditioner(data_batch)
        condition.data_type = DataType.VIDEO

        return raw_state, latent_state, condition
