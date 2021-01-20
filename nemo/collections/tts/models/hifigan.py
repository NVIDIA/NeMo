
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

import torch
import torch.nn.functional as F
from pytorch_lightning.loggers.wandb import WandbLogger
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
import itertools
import wandb

from nemo.collections.tts.helpers.helpers import plot_spectrogram_to_numpy
from nemo.collections.tts.models.base import Vocoder
from nemo.core.classes.common import typecheck
from nemo.core.neural_types.elements import AudioSignal, MelSpectrogramType
from nemo.core.neural_types.neural_type import NeuralType
from nemo.collections.tts.modules.hifigan_modules import MultiScaleDiscriminator, MultiPeriodDiscriminator, Generator 
from nemo.utils import logging

class HifiGanModel(Vocoder):
    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        """ TODO
        """
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        self.audio_to_melspec_precessor = instantiate(cfg.preprocessor)
        self.generator = instantiate(cfg.generator)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

        self.sample_rate = self._cfg.preprocessor.sample_rate

    def configure_optimizers(self):
        optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            self._cfg.optim.lr,
            betas=[self._cfg.optim.adam_b1, self._cfg.optim.adam_b2]
        )
        optim_d = torch.optim.AdamW(
            itertools.chain(self.msd.parameters(), self.mpd.parameters()),
            self._cfg.optim.lr,
            betas=[self._cfg.optim.adam_b1, self._cfg.optim.adam_b2]
        )

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g,
            gamma=self._cfg.optim.lr_decay,
            last_epoch=-1  # TODO: adjust last_epoch in case we load a checkpoint
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optim_d,
            gamma=self._cfg.optim.lr_decay,
            last_epoch=-1  # TODO: adjust last_epoch in case we load a checkpoint
        )

        return [optim_g, optim_d], [scheduler_g, scheduler_d]

    @property
    def input_types(self):
        return {
            "spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "audio": NeuralType(('B', 'S', 'T'), AudioSignal(self.sample_rate)),
        }

    @typecheck()
    def forward(self, *, spec):
        """
        Runs the generator, for inputs and outputs see input_types, and output_types
        """
        return self.generator(x=spec)

    @typecheck(output_types={"audio": NeuralType(('B', 'T'), AudioSignal())})
    def convert_spectrogram_to_audio(self, spec: 'torch.tensor') -> 'torch.tensor':
        return self(spec=spec).squeeze(1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        audio, audio_len = batch
        audio_mel, _ = self.audio_to_melspec_precessor(audio, audio_len)
        audio_pred = self(spec=audio_mel)
        audio = audio.unsqueeze(1)

        # train generator
        if optimizer_idx == 0:
            audio_pred_mel, _ = self.audio_to_melspec_precessor(audio_pred.squeeze(1), audio_len)

            # L1 melspec loss
            loss_mel = F.l1_loss(audio_mel, audio_pred_mel) * 45

            # feature matching loss
            _, mpd_score_gen, fmap_mpd_real, fmap_mpd_gen = self.mpd(y=audio, y_hat=audio_pred)
            _, msd_score_gen, fmap_msd_real, fmap_msd_gen = self.msd(y=audio, y_hat=audio_pred)
            loss_fm_mpd = feature_loss(fmap_mpd_real, fmap_mpd_gen)
            loss_fm_msd = feature_loss(fmap_msd_real, fmap_msd_gen)
            loss_gen_mpd, _ = generator_loss(mpd_score_gen)
            loss_gen_msd, _ = generator_loss(msd_score_gen)
            loss_g = loss_gen_msd + loss_gen_mpd + loss_fm_msd + loss_fm_mpd + loss_mel

            metrics = {
                "g_l1_loss": loss_mel,
                "g_loss_fm_mpd": loss_fm_mpd,
                "g_loss_fm_msd": loss_fm_msd,
                "g_loss_gen_mpd": loss_gen_mpd,
                "g_loss_gen_msd": loss_gen_msd,
                "g_loss": loss_g
            }
            self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
            return loss_g
        
        # train discriminator
        else:
            mpd_score_real, mpd_score_gen, _, _ = self.mpd(y=audio, y_hat=audio_pred.detach())
            loss_disc_mpd, _, _ = discriminator_loss(mpd_score_real, mpd_score_gen)
            msd_score_real, msd_score_gen, _, _ = self.msd(y=audio, y_hat=audio_pred.detach())
            loss_disc_msd, _, _ = discriminator_loss(msd_score_real, msd_score_gen)
            loss_disc_all = loss_disc_msd + loss_disc_mpd

            metrics = {
                "d_loss_mpd": loss_disc_mpd,
                "d_loss_msd": loss_disc_msd,
                "d_loss": loss_disc_all
            }
            self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
            return loss_disc_all

    def validation_step(self, batch, batch_idx):
        audio, audio_len = batch
        audio_mel, _ = self.audio_to_melspec_precessor(audio, audio_len)
        audio_pred = self(spec=audio_mel)

        audio_pred_mel, _ = self.audio_to_melspec_precessor(audio_pred.squeeze(1), audio_len)
        loss_mel = F.l1_loss(audio_mel, audio_pred_mel)

        self.log("val_loss", loss_mel, prog_bar=True, sync_dist=True)

        # plot audio once per epoch
        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({
                "audio": [
                    wandb.Audio(
                        audio[0].data.cpu().numpy(),
                        caption="real audio",
                        sample_rate=self.sample_rate
                    ),
                    wandb.Audio(
                        audio_pred[0, 0].data.cpu().numpy(),
                        caption="generated audio",
                        sample_rate=self.sample_rate
                    )
                ],
                "specs": [
                    wandb.Image(
                        plot_spectrogram_to_numpy(audio_mel[0].data.cpu().numpy()),
                        caption="real audio"
                    ),
                    wandb.Image(
                        plot_spectrogram_to_numpy(audio_pred_mel[0].data.cpu().numpy()),
                        caption="generated audio"
                    )
                ],
            },
            commit=False)

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")
        if shuffle_should_be:
            if 'shuffle' not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg["dataloader_params"]):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif not shuffle_should_be and cfg.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="validation")

    @classmethod
    def list_available_models(cls) -> 'Optional[Dict[str, str]]':
        # TODO
        pass

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
