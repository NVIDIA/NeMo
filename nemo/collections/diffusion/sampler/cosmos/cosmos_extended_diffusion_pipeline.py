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
from statistics import NormalDist
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import torch
import torch.distributed
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from nemo.collections.diffusion.sampler.batch_ops import *
from nemo.collections.diffusion.sampler.conditioner import (
    BaseVideoCondition,
    DataType,
    Edify4Condition,
    VideoExtendCondition,
)
from nemo.collections.diffusion.sampler.context_parallel import cat_outputs_cp, split_inputs_cp
from nemo.collections.diffusion.sampler.edm.edm import EDMSDE, EDMSampler, EDMScaling
from nemo.collections.diffusion.sampler.res.res_sampler import COMMON_SOLVER_OPTIONS, RESSampler

# key to check if the video data is normalized or image data is converted to video data
# to avoid apply normalization or augment image dimension multiple times
# It is due to we do not have normalization and augment image dimension in the dataloader and move it to the model
IS_PREPROCESSED_KEY = "is_preprocessed"


class ExtendedDiffusionPipeline:
    """
    Diffusion pipeline for EDM sampling.
    Currently only supports video diffusion inference.
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
        sampler_type="RES",  # or "RES"
        # EDM Scaling Args
        sigma_data=0.5,
        seed=1234,
        loss_add_logvar=True,
    ):
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
        noise_seed = self.seed
        noise_level_seed = self.seed
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
        output_batch, kendall_loss, pred_mse, edm_loss = self.compute_loss_with_epsilon_and_sigma(
            data_batch, x0_from_data_batch, x0, condition, epsilon, sigma
        )
        return output_batch, kendall_loss

    def denoise(
        self,
        xt: torch.Tensor,
        sigma: torch.Tensor,
        condition: VideoExtendCondition,
        condition_video_augment_sigma_in_inference: float = 0.001,
    ):
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

        xt = xt.to(**self.tensor_kwargs)
        sigma = sigma.to(**self.tensor_kwargs)

        gt_latent = condition.gt_latent
        condition_latent = gt_latent
        condition, augment_latent = self.augment_conditional_latent_frames(
            condition, condition_latent, condition_video_augment_sigma_in_inference, sigma
        )

        # Use xt and condition to get new_noise_xt
        condition_video_indicator = condition.condition_video_indicator  # [B, 1, T, 1, 1]
        if parallel_state.get_context_parallel_world_size() > 1:
            cp_group = parallel_state.get_context_parallel_group()
            condition_video_indicator = split_inputs_cp(condition_video_indicator, seq_dim=2, cp_group=cp_group)
            augment_latent = split_inputs_cp(augment_latent, seq_dim=2, cp_group=cp_group)
            gt_latent = split_inputs_cp(gt_latent, seq_dim=2, cp_group=cp_group)
            condition.condition_video_input_mask = split_inputs_cp(
                condition.condition_video_input_mask, seq_dim=2, cp_group=cp_group
            )
            if condition.condition_video_pose is not None:
                condition.condition_video_pose = split_inputs_cp(
                    condition.condition_video_pose, seq_dim=2, cp_group=cp_group
                )

        # Combine / concatenate the conditional video latent with the noisy input.
        new_noise_xt = condition_video_indicator * augment_latent + (1 - condition_video_indicator) * xt
        new_noise_xt = new_noise_xt.to(**self.tensor_kwargs)

        # get precondition for the network
        c_skip, c_out, c_in, c_noise = self.scaling(sigma=sigma)

        condition.data_type = DataType.VIDEO

        output = self.net(
            x=batch_mul(c_in, new_noise_xt),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps=c_noise,  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        )
        if isinstance(output, tuple):
            net_output, logvar = output
        else:
            net_output, logvar = output, None

        if not parallel_state.is_pipeline_last_stage():
            return net_output

        x0_pred = batch_mul(c_skip, new_noise_xt) + batch_mul(c_out, net_output)

        # get noise prediction based on sde
        eps_pred = batch_mul(new_noise_xt - x0_pred, 1.0 / sigma)

        x0_pred_replaced = condition_video_indicator * gt_latent + (1 - condition_video_indicator) * x0_pred
        x0_pred = x0_pred_replaced

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
        Compute loss given epsilon and sigma

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

        # make prediction
        x0_pred, eps_pred, logvar = self.denoise(xt, sigma, condition)
        # loss weights for different noise levels
        weights_per_sigma = self.get_per_sigma_loss_weights(sigma=sigma)
        # extra weight for each sample, for example, aesthetic weight, camera weight
        weights_per_sample = self.get_per_sample_weight(data_batch, x0.shape[0])
        # extra loss mask for each sample, for example, human faces, hands
        # loss_mask_per_sample = self.get_per_sample_loss_mask(data_batch, x0_from_data_batch.shape, x0.shape)
        loss_mask_per_sample = 1.0

        # Compute loss kernel.
        pred_mse = (x0 - x0_pred) ** 2 * loss_mask_per_sample  # Equation 5.

        edm_loss = batch_mul(pred_mse, weights_per_sigma * weights_per_sample)
        if len(edm_loss.shape) > 5:
            edm_loss = edm_loss.squeeze(0)
        b, c, t, h, w = edm_loss.shape
        if logvar is not None and self.loss_add_logvar:
            kendall_loss = batch_mul(edm_loss, torch.exp(-logvar).view(-1)).flatten(start_dim=1) + logvar.view(-1, 1)
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

    def get_per_sample_weight(self, data_batch: dict[str, torch.Tensor], batch_size: int):
        r"""
        extra weight for each sample, for example, aesthetic weight
        Args:
            data_batch: raw data batch draw from the training data loader.
            batch_size: int, the batch size of the input data
        """
        return 1.0

    def get_per_sample_loss_mask(self, data_batch, raw_x_shape, latent_x_shape):
        """
        extra loss mask for each sample, for example, human faces, hands.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            raw_x_shape (tuple): shape of the input data. We need the raw_x_shape for necessary resize operation.
            latent_x_shape (tuple): shape of the latent data
        """
        # if self.loss_mask_enabled:
        #     raw_x_shape = [raw_x_shape[0], 1, *raw_x_shape[2:]]
        #     weights = create_per_sample_loss_mask(
        #         self.loss_masking, data_batch, raw_x_shape, torch.get_default_dtype(), "cuda"
        #     )
        #     return F.interpolate(weights, size=latent_x_shape[2:], mode="bilinear")

        return 1.0

    def get_per_sigma_loss_weights(self, sigma: torch.Tensor):
        """
        Args:
            sigma (tensor): noise level

        Returns:
            loss weights per sigma noise level
        """
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    def get_x0_fn_from_batch_with_condition_latent(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        condition_latent: torch.Tensor = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
    ) -> Callable:
        """Creates a denoising function that incorporates latent conditioning for video generation.

        This function extends the base model's denoising by adding latent conditioning. It processes
        the input batch to create conditioned and unconditioned states, then returns a function that
        performs denoising with classifier-free guidance.

        Args:
            data_batch: Input data dictionary for conditioning
            guidance: Classifier-free guidance scale (default: 1.5)
            is_negative_prompt: Whether to use negative T5 prompt for unconditioned generation
            condition_latent: Video latent tensor of shape (B,C,T,H,W) used as condition
            num_condition_t: Number of timesteps to condition on when using "first_n" conditioning
            condition_video_augment_sigma_in_inference: Noise level for augmenting condition video
            add_input_frames_guidance: Whether to apply classifier-free guidance to input frames

        Returns:
            A function that takes noisy input x and noise level sigma and returns denoised prediction
        """
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        condition.video_cond_bool = True
        condition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, condition, num_condition_t
        )

        if 'plucker_embeddings' in data_batch:
            condition.add_pose_condition = True
            condition = self.add_condition_pose(data_batch, condition)

        uncondition.video_cond_bool = False if add_input_frames_guidance else True
        uncondition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, uncondition, num_condition_t
        )

        if condition.add_pose_condition:
            uncondition = self.add_condition_pose(data_batch, uncondition)

        condition_video_input_mask_copy = condition.condition_video_input_mask.clone().detach()
        uncondition_video_input_mask_copy = uncondition.condition_video_input_mask.clone().detach()

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            condition.condition_video_input_mask = condition_video_input_mask_copy.clone().detach()
            cond_x0_pred, _, _ = self.denoise(
                noise_x,
                sigma,
                condition,
                condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            )
            uncondition.condition_video_input_mask = uncondition_video_input_mask_copy.clone().detach()
            uncond_x0_pred, _, _ = self.denoise(
                noise_x,
                sigma,
                uncondition,
                condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            )

            return cond_x0_pred + guidance * (cond_x0_pred - uncond_x0_pred)

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

        x0_fn = self.get_x0_fn_from_batch_with_condition_latent(
            data_batch,
            guidance,
            is_negative_prompt=is_negative_prompt,
            condition_latent=condition_latent,
            num_condition_t=num_condition_t,
            condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            add_input_frames_guidance=add_input_frames_guidance,
        )

        state_shape = list(state_shape)
        np.random.seed(self.seed)
        x_sigma_max = (
            torch.from_numpy(np.random.randn(1, *state_shape).astype(np.float32)).to(
                dtype=torch.float32, device=self.tensor_kwargs["device"]
            )
            * self.sde.sigma_max
        )

        cp_enabled = parallel_state.get_context_parallel_world_size() > 1

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

    def draw_training_sigma_and_epsilon(self, x0_size: int, condition: Any) -> torch.Tensor:
        del condition
        batch_size = x0_size[0]
        if self._noise_generator is None:
            self._initialize_generators()
        epsilon = torch.randn(x0_size, **self.tensor_kwargs, generator=self._noise_generator)
        self.video_noise_multiplier = 1.0
        sigma_B = self.sde.sample_t(batch_size) * self.video_noise_multiplier
        return sigma_B.to(**self.tensor_kwargs), epsilon

    def draw_augment_sigma_and_epsilon(
        self, size: int, condition: VideoExtendCondition, p_mean: float, p_std: float, multiplier: float
    ) -> Tensor:
        del condition
        batch_size = size[0]
        epsilon = torch.randn(size, **self.tensor_kwargs)

        gaussian_dist = NormalDist(mu=p_mean, sigma=p_std)
        cdf_vals = np.random.uniform(size=(batch_size))
        samples_interval_gaussian = [gaussian_dist.inv_cdf(cdf_val) for cdf_val in cdf_vals]

        log_sigma = torch.tensor(samples_interval_gaussian, device="cuda")
        sigma_B = torch.exp(log_sigma) * multiplier

        return sigma_B.to(**self.tensor_kwargs), epsilon

    def get_data_and_condition(
        self,
        data_batch: dict[str, Tensor],
        num_condition_t: Union[int, None] = None,
    ) -> Tuple[Tensor, BaseVideoCondition]:
        # Latent state
        raw_state = data_batch["video"] * self.sigma_data
        # assume data is already encoded
        latent_state = raw_state
        # latent_state = self.encode(raw_state)

        # Condition
        condition = self.conditioner(data_batch)
        condition.data_type = DataType.VIDEO

        condition = self.add_condition_video_indicator_and_video_input_mask(
            latent_state, condition, num_condition_t=data_batch["num_condition_t"]
        )

        if 'plucker_embeddings' in data_batch:
            condition.add_pose_condition = True
            condition.first_random_n_num_condition_t_max = 1
            condition = self.add_condition_pose(data_batch, condition)

        return raw_state, latent_state, condition

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
        if condition.condition_location == "first_n":
            # Only in inference to decide the condition region
            assert num_condition_t is not None, "num_condition_t should be provided"
            assert num_condition_t <= T, f"num_condition_t should be less than T, get {num_condition_t}, {T}"
            condition_video_indicator[:, :, :num_condition_t] += 1.0
        elif condition.condition_location == "first_random_n":
            # Only in training
            num_condition_t_max = condition.first_random_n_num_condition_t_max
            assert (
                num_condition_t_max <= T
            ), f"num_condition_t_max should be less than T, get {num_condition_t_max}, {T}"
            assert num_condition_t_max >= condition.first_random_n_num_condition_t_min
            num_condition_t = torch.randint(
                condition.first_random_n_num_condition_t_min,
                num_condition_t_max + 1,
                (1,),
            ).item()
            condition_video_indicator[:, :, :num_condition_t] += 1.0

        elif condition.condition_location == "random":
            # Only in training
            condition_rate = condition.random_conditon_rate
            flag = torch.ones(1, 1, T, 1, 1, device=latent_state.device).type(latent_dtype) * condition_rate
            condition_video_indicator = torch.bernoulli(flag).type(latent_dtype).to(latent_state.device)
        else:
            raise NotImplementedError(
                f"condition_location {condition.condition_location} not implemented; training={self.training}"
            )
        condition.gt_latent = latent_state
        condition.condition_video_indicator = condition_video_indicator

        B, C, T, H, W = latent_state.shape
        # Create additional input_mask channel, this will be concatenated to the input of the network
        # See design doc section (Implementation detail A.1 and A.2) for visualization
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

    def augment_conditional_latent_frames(
        self,
        condition: VideoExtendCondition,
        gt_latent: Tensor,
        condition_video_augment_sigma_in_inference: float = 0.001,
        sigma: Tensor = None,
    ) -> Union[VideoExtendCondition, Tensor]:
        """This function is used to augment the condition input with noise
        Args:
            condition (VideoExtendCondition): condition object
                condition_video_indicator: binary tensor indicating the region is condition(value=1) or generation(value=0). Bx1xTx1x1 tensor.
                condition_video_input_mask: input mask for the network input, indicating the condition region. B,1,T,H,W tensor. will be concat with the input for the network.
            cfg_video_cond_bool (VideoCondBoolConfig): video condition bool config
            gt_latent (Tensor): ground truth latent tensor in shape B,C,T,H,W
            condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
            sigma (Tensor): noise level for the generation region
        Returns:
            VideoExtendCondition: updated condition object
                condition_video_augment_sigma: sigma for the condition region, feed to the network
            augment_latent (Tensor): augmented latent tensor in shape B,C,T,H,W

        """

        if condition.apply_corruption_to_condition_region == "noise_with_sigma":
            # Training only, sample sigma for the condition region
            augment_sigma, _ = self.draw_augment_sigma_and_epsilon(
                gt_latent.shape,
                condition,
                condition.augment_sigma_sample_p_mean,
                condition.augment_sigma_sample_p_std,
                condition.augment_sigma_sample_multiplier,
            )

        elif condition.apply_corruption_to_condition_region == "noise_with_sigma_fixed":
            # Inference only, use fixed sigma for the condition region
            assert (
                condition_video_augment_sigma_in_inference is not None
            ), "condition_video_augment_sigma_in_inference should be provided"
            augment_sigma = condition_video_augment_sigma_in_inference

            if augment_sigma >= sigma.flatten()[0]:
                # This is a inference trick! If the sampling sigma is smaller than the augment sigma, we will start denoising the condition region together.
                # This is achieved by setting all region as `generation`, i.e. value=0
                condition.condition_video_indicator = condition.condition_video_indicator * 0

            augment_sigma = torch.tensor([augment_sigma], **self.tensor_kwargs)

        else:
            raise ValueError(f"does not support {condition.apply_corruption_to_condition_region}")

        # Now apply the augment_sigma to the gt_latent
        noise = torch.randn(*gt_latent.shape, **self.tensor_kwargs)

        augment_latent = gt_latent + noise * augment_sigma[:, None, None, None, None]

        _, _, c_in_augment, c_noise_augment = self.scaling(sigma=augment_sigma)

        if condition.condition_on_augment_sigma:  # model takes augment_sigma as input
            if condition.condition_video_indicator.sum() > 0:  # has condition frames
                condition.condition_video_augment_sigma = c_noise_augment
            else:  # no condition frames
                condition.condition_video_augment_sigma = torch.zeros_like(c_noise_augment)

        # Multiply the whole latent with c_in_augment
        augment_latent_cin = batch_mul(augment_latent, c_in_augment)

        # Since the whole latent will multiply with c_in later, we divide the value to cancel the effect
        _, _, c_in, _ = self.scaling(sigma=sigma)
        augment_latent_cin = batch_mul(augment_latent_cin, 1 / c_in)

        return condition, augment_latent_cin

    def add_condition_pose(self, data_batch: Dict, condition: VideoExtendCondition) -> VideoExtendCondition:
        """Add pose condition to the condition object. For camera control model
        Args:
            data_batch (Dict): data batch, with key "plucker_embeddings", in shape B,T,C,H,W
            latent_state (torch.Tensor): latent state tensor in shape B,C,T,H,W
            condition (VideoExtendCondition): condition object
            num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        Returns:
            VideoExtendCondition: updated condition object
        """
        assert (
            "plucker_embeddings" in data_batch or "plucker_embeddings_downsample" in data_batch.keys()
        ), f"plucker_embeddings should be in data_batch. only find {data_batch.keys()}"
        plucker_embeddings = (
            data_batch["plucker_embeddings"]
            if "plucker_embeddings_downsample" not in data_batch.keys()
            else data_batch["plucker_embeddings_downsample"]
        )
        condition.condition_video_pose = rearrange(plucker_embeddings, "b t c h w -> b c t h w").contiguous()
        condition.condition_video_pose = condition.condition_video_pose.to(**self.tensor_kwargs)

        return condition
