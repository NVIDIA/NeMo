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

import math
from random import random, randrange
from typing import Dict, List, Optional, TypedDict, Union

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch import Tensor, nn
from torch.autograd import grad as torch_grad
from torch.utils.tensorboard.writer import SummaryWriter

import nemo
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.modules.spectrogram_enhancer import Discriminator, Generator, mask
from nemo.core import Exportable, ModelPT
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


def gradient_penalty(images, output, weight=10):
    batch_size = images.shape[0]
    gradients = torch_grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def calc_pl_lengths(styles, images):
    device = images.device
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape, device=device) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(
        outputs=outputs,
        inputs=styles,
        grad_outputs=torch.ones(outputs.shape, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()


def gen_hinge_loss(fake):
    return fake.mean()


def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()


def consistency_loss(condition, output, lengths):
    *_, w, h = condition.shape
    w, h = w // 4, h

    condition = torch.nn.functional.interpolate(condition, size=(w, h), mode="bilinear", antialias=True)
    output = torch.nn.functional.interpolate(output, size=(w, h), mode="bilinear", antialias=True)

    dist = (condition - output).abs()
    dist = mask(dist, lengths)
    return (dist / rearrange(lengths, "b 1 -> b 1 1 1")).sum(dim=-1).mean()


class SpectrogramEnhancerModel(ModelPT, Exportable):
    """GAN-based model to add details to blurry spectrograms from TTS models like Tacotron or FastPitch."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        self.spectrogram_model = None
        super().__init__(cfg=cfg, trainer=trainer)

        self.init_spectrogram_model()

        self.G = Generator(
            n_bands=cfg["n_bands"],
            latent_dim=cfg["latent_dim"],
            network_capacity=cfg["network_capacity"],
            style_depth=cfg["style_depth"],
            channels=1,
            fmap_max=cfg["fmap_max"],
        )
        self.D = Discriminator(
            n_bands=cfg["n_bands"], network_capacity=cfg["network_capacity"], channels=1, fmap_max=cfg["fmap_max"],
        )

    def init_spectrogram_model(self):
        if (path := self._cfg.get("spectrogram_model_path")) :
            self.spectrogram_model = SpectrogramGenerator.restore_from(path, map_location=torch.device("cpu"))
            self.spectrogram_model.freeze()

            self._cfg.train_ds = OmegaConf.merge(self.spectrogram_model._cfg.train_ds, self._cfg.train_ds)
            self.setup_training_data(self._cfg.train_ds)

    def move_to_correct_device(self, e):
        return type_as_recursive(e, next(iter(self.G.parameters())))

    def normalize_spectrograms(self, spectrogram: Tensor) -> Tensor:
        spectrogram = spectrogram - self._cfg["spectrogram_min_value"]
        spectrogram = spectrogram / (self._cfg["spectrogram_max_value"] - self._cfg["spectrogram_min_value"])
        return spectrogram

    def unnormalize_spectrograms(self, spectrogram: Tensor) -> Tensor:
        spectrogram = spectrogram * (self._cfg["spectrogram_max_value"] - self._cfg["spectrogram_min_value"])
        spectrogram = spectrogram + self._cfg["spectrogram_min_value"]
        return spectrogram

    def generate_zs(self, batch_size: int = 1, mixing: bool = False):
        if mixing and self._cfg["mixed_prob"] < random():
            mixing_point = randrange(1, self.G.num_layers)
            first_part = [torch.randn(batch_size, self._cfg["latent_dim"])] * mixing_point
            second_part = [torch.randn(batch_size, self._cfg["latent_dim"])] * (self.G.num_layers - mixing_point)
            zs = [*first_part, *second_part]
        else:
            zs = [torch.randn(batch_size, self._cfg["latent_dim"])] * self.G.num_layers

        return self.move_to_correct_device(zs)

    def generate_noise(self, batch_size: int = 1) -> Tensor:
        noise = torch.rand(batch_size, self._cfg["n_bands"], 4096, 1)
        return self.move_to_correct_device(noise)

    def pad_spectrogram(self, spectrogram):
        multiplier = 2 ** sum(1 for block in self.G.blocks if block.upsample)
        *_, max_length = spectrogram.shape
        return torch.nn.functional.pad(spectrogram, (0, multiplier - max_length % multiplier))

    def forward(
        self,
        condition: Tensor,
        lengths: Tensor,
        zs: Optional[List[Tensor]] = None,
        ws: Optional[List[Tensor]] = None,
        noise: Optional[Tensor] = None,
        mixing: bool = False,
        normalize: bool = True,
    ):
        if len(condition.shape) != 4:
            raise ValueError(f"Got spectrogram tensor of shape {condition.shape}, expected B x 1 x C x L")

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

        return enhanced_spectrograms

    def prepare_batch(self, batch):
        if not isinstance(self.spectrogram_model, FastPitchModel):
            raise NotImplementedError(
                f"{type(self.spectrogram_model)} is not supported, please implement batch preparation for this model."
            )

        attn_prior, durs, speaker = None, None, None
        (audio, audio_lens, text, text_lens, attn_prior, pitch, _, speaker,) = batch
        mels, mel_lens = self.spectrogram_model.preprocessor(input_signal=audio, length=audio_lens)

        mels_pred, *_, pitch = self.spectrogram_model(
            text=text,
            durs=durs,
            pitch=pitch,
            speaker=speaker,
            pace=1.0,
            spec=mels if self.spectrogram_model.learn_alignment else None,
            attn_prior=attn_prior,
            mel_lens=mel_lens,
            input_lens=text_lens,
        )

        mels = mels.cpu()
        mel_lens = mel_lens.cpu()
        mels_pred = mels_pred.cpu()

        batch_size, *_ = mels.shape

        targets = []
        conditions = []
        lengths = []

        for i in range(batch_size):
            l = mel_lens[i].item()
            lengths.append(l)

            targets.append(mels[i, :, :l].T)
            conditions.append(mels_pred[i, :, :l].T)

        target = nn.utils.rnn.pad_sequence(targets, batch_first=True)
        condition = nn.utils.rnn.pad_sequence(conditions, batch_first=True)
        lengths = torch.LongTensor(lengths).unsqueeze(-1)

        target = rearrange(target, "b l c -> b 1 c l")
        condition = rearrange(condition, "b l c -> b 1 c l")

        return self.move_to_correct_device((target, condition, lengths))

    def training_step(self, batch, batch_idx, optimizer_idx):
        target, condition, lengths = self.prepare_batch(batch)

        with torch.no_grad():
            target = self.normalize_spectrograms(target)
            condition = self.normalize_spectrograms(condition)

            target = mask(target, lengths)
            condition = mask(condition, lengths)

        # train discriminator
        if optimizer_idx == 0:
            enhanced_spectrograms = self.forward(condition, lengths, mixing=True, normalize=False)
            fake_logits = self.D(enhanced_spectrograms, condition, lengths)

            real_images = target.requires_grad_()
            real_logits = self.D(real_images, condition, lengths)
            d_loss = hinge_loss(real=real_logits, fake=fake_logits)
            self.log("d_loss", d_loss, prog_bar=True)

            if batch_idx % 4 == 0:
                gp_loss = gradient_penalty(real_images, real_logits)
                self.log("d_loss_gp", gp_loss, prog_bar=True)
                return d_loss + gp_loss

            return d_loss

        # train generator
        if optimizer_idx == 1:
            enhanced_spectrograms = self.forward(condition, lengths, mixing=True, normalize=False)
            fake_logits = self.D(enhanced_spectrograms, condition, lengths)
            g_loss = gen_hinge_loss(fake_logits)
            c_loss = 10 * consistency_loss(condition, enhanced_spectrograms, lengths)

            self.log("g_loss", g_loss, prog_bar=True)
            self.log("c_loss", c_loss, prog_bar=True)

            with torch.no_grad():
                self.log_illustration(target, condition, enhanced_spectrograms, lengths)
            return g_loss + c_loss

    def configure_optimizers(self):
        generator_opt = torch.optim.Adam(self.G.parameters(), lr=self._cfg["lr"], betas=(0.5, 0.9),)
        discriminator_opt = torch.optim.Adam(self.D.parameters(), lr=self._cfg["lr"], betas=(0.5, 0.9))
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
        return None

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
