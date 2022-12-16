# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# The following is largely based on code from https://github.com/lucidrains/stylegan2-pytorch

from random import random, randrange
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch import Tensor, nn
from torch.autograd import grad as torch_grad
from torch.utils.tensorboard.writer import SummaryWriter

from nemo.collections.tts.helpers.helpers import process_batch
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.modules.spectrogram_enhancer import Discriminator, Generator, mask
from nemo.core import Exportable, ModelPT, typecheck
from nemo.core.neural_types import LengthsType, MelSpectrogramType, NeuralType
from nemo.core.neural_types.elements import BoolType
from nemo.utils import logging


def type_as_recursive(e, source: Tensor):
    if isinstance(e, (list, tuple)):
        return [type_as_recursive(elem, source) for elem in e]
    elif isinstance(e, dict):
        return {key: type_as_recursive(value, source) for key, value in e.items()}
    elif isinstance(e, Tensor):
        return e.type_as(source)
    else:
        return e


class GradientPenaltyLoss(nn.Module):
    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.weight = weight

    def __call__(self, images, output):
        batch_size, *_ = images.shape
        gradients = torch_grad(
            outputs=output,
            inputs=images,
            grad_outputs=torch.ones(output.size(), device=images.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.reshape(batch_size, -1)
        return self.weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


class GeneratorLoss(nn.Module):
    def __call__(self, fake_logits):
        return fake_logits.mean()


class HingeLoss(nn.Module):
    def __call__(self, real_logits, fake_logits):
        return (F.relu(1 + real_logits) + F.relu(1 - fake_logits)).mean()


class ConsistencyLoss(nn.Module):
    def __init__(self, weight: float = 10):
        super().__init__()
        self.weight = weight

    def __call__(self, condition, output, lengths):
        *_, w, h = condition.shape
        w, h = w // 4, h

        condition = torch.nn.functional.interpolate(condition, size=(w, h), mode="bilinear", antialias=True)
        output = torch.nn.functional.interpolate(output, size=(w, h), mode="bilinear", antialias=True)

        dist = (condition - output).abs()
        dist = mask(dist, lengths)
        return (dist / rearrange(lengths, "b -> b 1 1 1")).sum(dim=-1).mean() * self.weight


class SpectrogramEnhancerModel(ModelPT, Exportable):
    """
    GAN-based model to add details to blurry spectrograms from TTS models like Tacotron or FastPitch. Based on StyleGAN 2 [1]
    [1] Karras et. al. - Analyzing and Improving the Image Quality of StyleGAN (https://arxiv.org/abs/1912.04958)
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        self.spectrogram_model = None
        super().__init__(cfg=cfg, trainer=trainer)

        self.init_spectrogram_model()

        self.G = instantiate(cfg.generator)
        self.D = instantiate(cfg.discriminator)

        self.generator_loss = GeneratorLoss()
        self.discriminator_loss = HingeLoss()
        self.consistency_loss = ConsistencyLoss(cfg.consistency_loss_weight)
        self.gradient_penalty_loss = GradientPenaltyLoss(cfg.gradient_penalty_loss_weight)

    def init_spectrogram_model(self):
        if (path := self._cfg.get("spectrogram_model_path")) :
            self.spectrogram_model = SpectrogramGenerator.restore_from(path, map_location=torch.device("cpu"))
            self.spectrogram_model.freeze()

            self._cfg.train_ds = OmegaConf.merge(self.spectrogram_model._cfg.train_ds, self._cfg.train_ds)
            self.setup_training_data(self._cfg.train_ds)

    def move_to_correct_device(self, e):
        return type_as_recursive(e, next(iter(self.G.parameters())))

    def normalize_spectrograms(self, spectrogram: Tensor) -> Tensor:
        spectrogram = spectrogram - self._cfg.spectrogram_min_value
        spectrogram = spectrogram / (self._cfg.spectrogram_max_value - self._cfg.spectrogram_min_value)
        return spectrogram

    def unnormalize_spectrograms(self, spectrogram: Tensor) -> Tensor:
        spectrogram = spectrogram * (self._cfg.spectrogram_max_value - self._cfg.spectrogram_min_value)
        spectrogram = spectrogram + self._cfg.spectrogram_min_value
        return spectrogram

    def generate_zs(self, batch_size: int = 1, mixing: bool = False):
        if mixing and self._cfg.mixed_prob < random():
            mixing_point = randrange(1, self.G.num_layers)
            first_part = [torch.randn(batch_size, self._cfg.latent_dim)] * mixing_point
            second_part = [torch.randn(batch_size, self._cfg.latent_dim)] * (self.G.num_layers - mixing_point)
            zs = [*first_part, *second_part]
        else:
            zs = [torch.randn(batch_size, self._cfg.latent_dim)] * self.G.num_layers

        return self.move_to_correct_device(zs)

    def generate_noise(self, batch_size: int = 1) -> Tensor:
        noise = torch.rand(batch_size, self._cfg.n_bands, 4096, 1)
        return self.move_to_correct_device(noise)

    def pad_spectrogram(self, spectrogram):
        multiplier = 2 ** sum(1 for block in self.G.blocks if block.upsample)
        *_, max_length = spectrogram.shape
        return torch.nn.functional.pad(spectrogram, (0, multiplier - max_length % multiplier))

    @typecheck(
        input_types={
            "condition": NeuralType(("B", "D", "T_spec"), MelSpectrogramType()),
            "lengths": NeuralType(("B"), LengthsType()),
            "mixing": NeuralType(None, BoolType(), optional=True),
            "normalize": NeuralType(None, BoolType(), optional=True),
        }
    )
    def forward(
        self, *, condition: Tensor, lengths: Tensor, mixing: bool = False, normalize: bool = True,
    ):
        """
        Generator forward pass. Noise inputs will be generated.

        condition: batch of blurred spectrograms
        lenghts: length for every spectrogam in the batch
        mixing: style mixing, usually True during training
        normalize: normalize spectrogram range to ~[0, 1], True for normal use

        For explanation of style mixing refer to [1]
        [1] Karras et. al. - A Style-Based Generator Architecture for Generative Adversarial Networks, 2018 (https://arxiv.org/abs/1812.04948)
        """

        return self.forward_with_custom_noise(
            condition=condition, lengths=lengths, mixing=mixing, normalize=normalize, zs=None, ws=None, noise=None
        )

    def forward_with_custom_noise(
        self,
        condition: Tensor,
        lengths: Tensor,
        zs: Optional[List[Tensor]] = None,
        ws: Optional[List[Tensor]] = None,
        noise: Optional[Tensor] = None,
        mixing: bool = False,
        normalize: bool = True,
    ):
        """
        Generator forward pass. Noise inputs will be generated if None.

        condition: batch of blurred spectrograms
        lenghts: length for every spectrogam in the batch
        zs: latent noise inputs on the unit sphere (either this or ws or neither)
        ws: latent noise inputs in the style space (either this or zs or neither)
        noise: per-pixel indepentent gaussian noise
        mixing: style mixing, usually True during training
        normalize: normalize spectrogram range to ~[0, 1], True for normal use

        For explanation of style mixing refer to [1]
        For definititions of z, w [2]
        [1] Karras et. al. - A Style-Based Generator Architecture for Generative Adversarial Networks, 2018 (https://arxiv.org/abs/1812.04948)
        [2] Karras et. al. - Analyzing and Improving the Image Quality of StyleGAN, 2019 (https://arxiv.org/abs/1912.04958)
        """
        batch_size, *_, max_length = condition.shape

        # generate noise
        if zs is not None and ws is not None:
            raise ValueError(
                "Please specify either zs or ws or neither, but not both. It is not clear which one to use."
            )

        if zs is None:
            zs = self.generate_zs(batch_size, mixing)
        if ws is None:
            ws = [self.G.style_mapping(z) for z in zs]
        if noise is None:
            noise = self.generate_noise(batch_size)

        condition = rearrange(condition, "b c l -> b 1 c l")
        # normalize if needed, mask and pad appropriately
        if normalize:
            condition = self.normalize_spectrograms(condition)
        condition = self.pad_spectrogram(condition)
        condition = mask(condition, lengths)

        # the main call
        enhanced_spectrograms = self.G(condition, lengths, ws, noise)

        # denormalize if needed, mask and remove padding
        if normalize:
            enhanced_spectrograms = self.unnormalize_spectrograms(enhanced_spectrograms)
        enhanced_spectrograms = mask(enhanced_spectrograms, lengths)
        enhanced_spectrograms = enhanced_spectrograms[:, :, :, :max_length]
        enhanced_spectrograms = rearrange(enhanced_spectrograms, "b 1 c l -> b c l")

        return enhanced_spectrograms

    @torch.no_grad()
    def prepare_batch(self, batch):
        if not isinstance(self.spectrogram_model, FastPitchModel):
            raise NotImplementedError(
                f"{type(self.spectrogram_model)} is not supported, please implement batch preparation for this model."
            )

        batch = process_batch(batch, self._train_dl.dataset.sup_data_types)
        mels, mel_lens = self.spectrogram_model.preprocessor(input_signal=batch["audio"], length=batch["audio_lens"])

        mels_pred, *_ = self.spectrogram_model(
            text=batch["text"],
            durs=None,
            pitch=batch["pitch"],
            speaker=batch.get("speaker"),
            pace=1.0,
            spec=mels if self.spectrogram_model.learn_alignment else None,
            attn_prior=batch.get("attn_prior"),
            mel_lens=mel_lens,
            input_lens=batch["text_lens"],
        )

        max_len = mel_lens.max().item()
        target = mels[:, :, :max_len]
        condition = mels_pred[:, :, :max_len]

        return target, condition, mel_lens

    def training_step(self, batch, batch_idx, optimizer_idx):
        target, condition, lengths = self.prepare_batch(batch)

        with torch.no_grad():
            target = self.normalize_spectrograms(target)
            condition = self.normalize_spectrograms(condition)

        # train discriminator
        if optimizer_idx == 0:
            enhanced_spectrograms = self.forward(condition=condition, lengths=lengths, mixing=True, normalize=False)
            enhanced_spectrograms = rearrange(enhanced_spectrograms, "b c l -> b 1 c l")
            fake_logits = self.D(enhanced_spectrograms, condition, lengths)

            real_images = rearrange(target, "b c l -> b 1 c l")
            real_images = real_images.requires_grad_()
            real_logits = self.D(real_images, condition, lengths)
            d_loss = self.discriminator_loss(real_logits, fake_logits)
            self.log("d_loss", d_loss, prog_bar=True)

            if batch_idx % self._cfg.gradient_penalty_loss_every_n_steps == 0:
                gp_loss = self.gradient_penalty_loss(real_images, real_logits)
                self.log("d_loss_gp", gp_loss, prog_bar=True)
                return d_loss + gp_loss

            return d_loss

        # train generator
        if optimizer_idx == 1:
            enhanced_spectrograms = self.forward(condition=condition, lengths=lengths, mixing=True, normalize=False)
            enhanced_spectrograms = rearrange(enhanced_spectrograms, "b c l -> b 1 c l")
            condition = rearrange(condition, "b c l -> b 1 c l")
            fake_logits = self.D(enhanced_spectrograms, condition, lengths)
            g_loss = self.generator_loss(fake_logits)
            c_loss = self.consistency_loss(condition, enhanced_spectrograms, lengths)

            self.log("g_loss", g_loss, prog_bar=True)
            self.log("c_loss", c_loss, prog_bar=True)

            with torch.no_grad():
                target = rearrange(target, "b c l -> b 1 c l")
                self.log_illustration(target, condition, enhanced_spectrograms, lengths)
            return g_loss + c_loss

    def configure_optimizers(self):
        generator_opt = instantiate(self._cfg.generator_opt, params=self.G.parameters(),)
        discriminator_opt = instantiate(self._cfg.discriminator_opt, params=self.D.parameters())
        return [discriminator_opt, generator_opt], []

    def setup_training_data(self, train_data_config):
        if self.spectrogram_model:
            self.spectrogram_model.setup_training_data(train_data_config)
            self._train_dl = self.spectrogram_model._train_dl

    def setup_validation_data(self, val_data_config):
        if self.spectrogram_model and val_data_config:
            self.spectrogram_model.setup_validation_data(val_data_config)
            self._validation_dl = self.spectrogram_model._validation_dl

    @classmethod
    def list_available_models(cls):
        return []

    def save_to(self, save_path: str):
        # when saving this model for further use in a .nemo file, we do not care about TTS model used to train it
        if self.spectrogram_model:
            spectrogram_model = self._modules.pop("spectrogram_model")
            cfg = DictConfig.copy(self._cfg)
            OmegaConf.set_struct(self._cfg, False)
            self._cfg.pop("spectrogram_model_path")
            OmegaConf.set_struct(self._cfg, True)

            super().save_to(save_path)

            self.spectrogram_model = spectrogram_model
            self._cfg = cfg
        else:
            super().save_to(save_path)

    def log_illustration(self, target, condition, enhanced, lengths):
        if self.global_rank != 0:
            return

        if not self.loggers:
            return

        step = self.trainer.global_step // 2  # because of G/D training
        if step % self.trainer.log_every_n_steps != 0:
            return

        idx = 0
        length = int(lengths.flatten()[idx].item())
        tensor = torch.stack([enhanced - condition, condition, enhanced, target], dim=0).cpu()[:, idx, :, :, :length]

        grid = torchvision.utils.make_grid(tensor, nrow=1).clamp(0.0, 1.0)

        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                writer: SummaryWriter = logger.experiment
                writer.add_image("spectrograms", grid, global_step=step)
                writer.flush()
            elif isinstance(logger, WandbLogger):
                logger.log_image("spectrograms", [grid], caption=["residual, input, output, ground truth"], step=step)
            else:
                logging.warning("Unsupported logger type: %s", str(type(logger)))
