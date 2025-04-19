# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Implements the forward op for training, validation, and inference."""

import os
from typing import Optional

import lightning.pytorch as L
import torch
import wandb
from cosmos1.models.tokenizer.networks.configs import continuous_video, discrete_video
from cosmos1.models.tokenizer.networks.continuous_video import CausalContinuousVideoTokenizer
from cosmos1.models.tokenizer.networks.discrete_video import CausalDiscreteVideoTokenizer
from einops import rearrange

from nemo.collections.llm import fn
from nemo.collections.physicalai.tokenizer.losses.config import VideoLoss
from nemo.collections.physicalai.tokenizer.losses.loss import TokenizerLoss
from nemo.lightning import io
from nemo.lightning.pytorch.optim import OptimizerModule

IMAGE_KEY = "images"
INPUT_KEY = "INPUT"
MASK_KEY = "loss_mask"
RECON_KEY = "reconstructions"
VIDEO_KEY = "video"

RECON_CONSISTENCY_KEY = f"{RECON_KEY}_consistency"
VIDEO_CONSISTENCY_LOSS = "video_consistency"

PREDICTION = "prediction"
EMA_PREDICTION = "ema_prediction"


class TokenizerModel(L.LightningModule, io.IOMixin, fn.FNMixin):
    def __init__(
        self,
        jit_ckpt_pth=None,
        model="Cosmos-1.0-Tokenizer-CV8x8x8",
        precision=torch.bfloat16,
        device=torch.device("cuda"),
        optim: Optional[OptimizerModule] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        autoencoder_path = os.path.join(jit_ckpt_pth, "autoencoder.jit")
        if not os.path.exists(autoencoder_path):
            raise FileNotFoundError(f"autoencoder.jit not found at {autoencoder_path}")

        weight = torch.jit.load(autoencoder_path)
        if model == "Cosmos-1.0-Tokenizer-CV8x8x8":
            self.network = CausalContinuousVideoTokenizer(**continuous_video)
        elif model == "Cosmos-1.0-Tokenizer-DV8x16x16":
            self.network = CausalDiscreteVideoTokenizer(**discrete_video)
        else:
            raise NotImplementedError(
                "We only support Cosmos-1.0-Tokenizer-CV8x8x8 and Cosmos-1.0-Tokenizer-DV8x16x16"
            )
        self.network.load_state_dict(weight.state_dict(), strict=False)
        self.network = self.network.to(device=device, dtype=precision)
        self.network.training = True

        self.loss = TokenizerLoss(config=VideoLoss())
        self.loss = self.loss.to(device=device, dtype=precision)

        self._precision = precision
        self._device = device

        self.image_key = IMAGE_KEY
        self.video_key = VIDEO_KEY

    def get_input_key(self, data_batch: dict[str, torch.Tensor]) -> str:
        if self.video_key in data_batch:
            return self.video_key
        else:
            raise ValueError("Input key not found in data_batch.")

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        self.network = self.network.to(device=self._device, dtype=self._precision, memory_format=memory_format)
        self.loss = self.loss.to(device=self._device, dtype=self._precision, memory_format=memory_format)

    def _network_forward(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Do the forward pass.
        tensor_batch = data_batch[self.get_input_key(data_batch)]
        output_batch = self.network(tensor_batch)
        output_batch = output_batch if self.network.training else output_batch._asdict()

        return output_batch

    def _training_step(
        self,
        data_batch: dict[str, torch.Tensor],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        _input_key = self.get_input_key(data_batch)
        output_dict = self._network_forward(data_batch)
        input_images, recon_images = data_batch[_input_key], output_dict[RECON_KEY]

        # pass loss_mask to loss computation
        inputs = {INPUT_KEY: input_images, MASK_KEY: data_batch.get("loss_mask", torch.ones_like(input_images))}
        loss_dict, loss_value = self.loss(inputs, output_dict, iteration)
        for k, v in loss_dict['loss'].items():
            self.log(k, v)
        self.log('loss', loss_value, prog_bar=True)
        self.log("global_step", self.global_step)

        return dict({PREDICTION: recon_images, **loss_dict}), loss_value

    def training_step(
        self,
        data_batch: dict[str, torch.Tensor],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        _, loss_value = self._training_step(data_batch, iteration)
        return loss_value

    # TODO: add validation step
    @torch.no_grad()
    def validation_step(
        self,
        data_batch: dict[str, torch.Tensor],
        iteration: int,
        ema_model: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        _input_key = self.get_input_key(data_batch)
        output_dict = self._network_forward(data_batch)
        input_images, recon_images = data_batch[_input_key], output_dict[RECON_KEY]

        # pass loss_mask to loss computation
        inputs = {INPUT_KEY: input_images, MASK_KEY: data_batch.get("loss_mask", torch.ones_like(input_images))}

        loss_dict, loss_value = self.loss(inputs, output_dict, iteration)

        if wandb.run is not None:

            visualization = torch.cat(
                [
                    rearrange(input_images[0], 'c t h w -> t c h w').cpu(),
                    rearrange(recon_images[0], 'c t h w -> t c h w').cpu(),
                ],
                axis=-1,
            )
            visualization = ((visualization + 0.5).clamp(0, 1).cpu().float().numpy() * 255).astype('uint8')

            wandb.log(
                {
                    "Original (left), Reconstruction (right)": [
                        wandb.Video(visualization, fps=24),
                    ]
                },
            )

        return loss_value

    @torch.inference_mode()
    def forward(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_dict = self._network_forward(data_batch)
        return dict({PREDICTION: output_dict[RECON_KEY]})
