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

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from cosmos1.models.diffusion.conditioner import CosmosCondition
from cosmos1.models.diffusion.diffusion.functional.batch_ops import batch_mul
from cosmos1.models.diffusion.diffusion.modules.denoiser_scaling import EDMScaling
from cosmos1.models.diffusion.diffusion.modules.res_sampler import COMMON_SOLVER_OPTIONS, Sampler
from cosmos1.models.diffusion.diffusion.types import DenoisePrediction
from cosmos1.models.diffusion.module.blocks import FourierFeatures
from cosmos1.models.diffusion.module.pretrained_vae import BaseVAE
from cosmos1.utils import log, misc
from cosmos1.utils.lazy_config import instantiate as lazy_instantiate


class EDMSDE:
    def __init__(
        self,
        sigma_max: float,
        sigma_min: float,
    ):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min


class DiffusionT2WModel(torch.nn.Module):
    """Text-to-world diffusion model that generates video frames from text descriptions.

    This model implements a diffusion-based approach for generating videos conditioned on text input.
    It handles the full pipeline including encoding/decoding through a VAE, diffusion sampling,
    and classifier-free guidance.
    """

    def __init__(self, config):
        """Initialize the diffusion model.

        Args:
            config: Configuration object containing model parameters and architecture settings
        """
        super().__init__()
        # Initialize trained_data_record with defaultdict, key: image, video, iteration
        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        log.debug(f"DiffusionModel: precision {self.precision}")
        # Timer passed to network to detect slow ranks.
        # 1. set data keys and data information
        self.sigma_data = config.sigma_data
        self.state_shape = list(config.latent_shape)
        self.setup_data_key()

        # 2. setup up diffusion processing and scaling~(pre-condition), sampler
        self.sde = EDMSDE(sigma_max=80, sigma_min=0.0002)
        self.sampler = Sampler()
        self.scaling = EDMScaling(self.sigma_data)
        self.tokenizer = None
        self.model = None

    @property
    def net(self):
        return self.model.net

    @property
    def conditioner(self):
        return self.model.conditioner

    @property
    def logvar(self):
        return self.model.logvar

    def set_up_tokenizer(self, tokenizer_dir: str):
        self.tokenizer: BaseVAE = lazy_instantiate(self.config.tokenizer)
        self.tokenizer.load_weights(tokenizer_dir)
        if hasattr(self.tokenizer, "reset_dtype"):
            self.tokenizer.reset_dtype()

    @misc.timer("DiffusionModel: set_up_model")
    def set_up_model(self, memory_format: torch.memory_format = torch.preserve_format):
        """Initialize the core model components including network, conditioner and logvar."""
        self.model = self.build_model()
        self.model = self.model.to(memory_format=memory_format, **self.tensor_kwargs)

    def build_model(self) -> torch.nn.ModuleDict:
        """Construct the model's neural network components.

        Returns:
            ModuleDict containing the network, conditioner and logvar components
        """
        config = self.config
        net = lazy_instantiate(config.net)
        conditioner = lazy_instantiate(config.conditioner)
        logvar = torch.nn.Sequential(
            FourierFeatures(num_channels=128, normalize=True), torch.nn.Linear(128, 1, bias=False)
        )

        return torch.nn.ModuleDict(
            {
                "net": net,
                "conditioner": conditioner,
                "logvar": logvar,
            }
        )

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encode input state into latent representation using VAE.

        Args:
            state: Input tensor to encode

        Returns:
            Encoded latent representation scaled by sigma_data
        """
        return self.tokenizer.encode(state) * self.sigma_data

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to pixel space using VAE.

        Args:
            latent: Latent tensor to decode

        Returns:
            Decoded tensor in pixel space
        """
        return self.tokenizer.decode(latent / self.sigma_data)

    def setup_data_key(self) -> None:
        """Configure input data keys for video and image data."""
        self.input_data_key = self.config.input_data_key  # by default it is video key for Video diffusion model

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function processes the input data batch through a conditioning workflow to obtain
        conditioned and unconditioned states. It then defines a nested function `x0_fn` which
        applies denoising on an input `noise_x` at a given noise level `sigma`.

        Args:
            data_batch: A batch of data used for conditioning. Format should align with conditioner
            guidance: Scalar value that modulates influence of conditioned vs unconditioned state
            is_negative_prompt: Use negative prompt t5 in uncondition if true

        Returns:
            A function `x0_fn(noise_x, sigma)` that takes noise_x and sigma, returns x0 prediction
        """
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(noise_x, sigma, condition).x0
            uncond_x0 = self.denoise(noise_x, sigma, uncondition).x0
            raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)
            if "guided_image" in data_batch:
                # replacement trick that enables inpainting with base model
                assert "guided_mask" in data_batch, "guided_mask should be in data_batch if guided_image is present"
                guide_image = data_batch["guided_image"]
                guide_mask = data_batch["guided_mask"]
                raw_x0 = guide_mask * guide_image + (1 - guide_mask) * raw_x0

            return raw_x0

        return x0_fn

    def denoise(self, xt: torch.Tensor, sigma: torch.Tensor, condition: CosmosCondition) -> DenoisePrediction:
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (CosmosCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred) and optional confidence (logvar).
        """

        xt = xt.to(**self.tensor_kwargs)
        sigma = sigma.to(**self.tensor_kwargs)
        # get precondition for the network
        c_skip, c_out, c_in, c_noise = self.scaling(sigma=sigma)

        # forward pass through the network
        net_output = self.net(
            x=batch_mul(c_in, xt),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps=c_noise,  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        )

        logvar = self.model.logvar(c_noise)
        x0_pred = batch_mul(c_skip, xt) + batch_mul(c_out, net_output)

        # get noise prediction based on sde
        eps_pred = batch_mul(xt - x0_pred, 1.0 / sigma)

        return DenoisePrediction(x0_pred, eps_pred, logvar)

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
        x_sigma_max: Optional[torch.Tensor] = None,
        sigma_max: float | None = None,
    ) -> Tensor:
        """Generate samples from a data batch using diffusion sampling.

        This function generates samples from either image or video data batches using diffusion sampling.
        It handles both conditional and unconditional generation with classifier-free guidance.

        Args:
            data_batch (Dict): Raw data batch from the training data loader
            guidance (float, optional): Classifier-free guidance weight. Defaults to 1.5.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
            state_shape (Tuple | None, optional): Shape of the state tensor. Uses self.state_shape if None. Defaults to None.
            n_sample (int | None, optional): Number of samples to generate. Defaults to None.
            is_negative_prompt (bool, optional): Whether to use negative prompt for unconditional generation. Defaults to False.
            num_steps (int, optional): Number of diffusion sampling steps. Defaults to 35.
            solver_option (COMMON_SOLVER_OPTIONS, optional): Differential equation solver option. Defaults to "2ab" (multistep solver).
            x_sigma_max (Optional[torch.Tensor], optional): Initial noisy tensor. If None, randomly initialized. Defaults to None.
            sigma_max (float | None, optional): Maximum noise level. Uses self.sde.sigma_max if None. Defaults to None.

        Returns:
            Tensor: Generated samples after diffusion sampling
        """
        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)
        if sigma_max is None:
            sigma_max = self.sde.sigma_max
        else:
            log.info("Using provided sigma_max for diffusion sampling.")
        if x_sigma_max is None:
            x_sigma_max = (
                misc.arch_invariant_rand(
                    (n_sample,) + tuple(state_shape),
                    torch.float32,
                    self.tensor_kwargs["device"],
                    seed,
                )
                * sigma_max
            )

        samples = self.sampler(
            x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=sigma_max, solver_option=solver_option
        )

        return samples
