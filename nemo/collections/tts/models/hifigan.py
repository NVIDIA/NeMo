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

import itertools

import torch
import torch.nn.functional as F
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.loggers.wandb import WandbLogger

from nemo.collections.tts.helpers.helpers import plot_spectrogram_to_numpy
from nemo.collections.tts.losses.hifigan_losses import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss
from nemo.collections.tts.models.base import Vocoder
from nemo.collections.tts.modules.hifigan_modules import MultiPeriodDiscriminator, MultiScaleDiscriminator
from nemo.core.classes.common import typecheck
from nemo.core.neural_types.elements import AudioSignal, MelSpectrogramType
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging


class HifiGanModel(Vocoder):
    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        self.audio_to_melspec_precessor = instantiate(cfg.preprocessor)
        # use a different melspec extractor because:
        # 1. we need to pass grads
        # 2. we need remove fmax limitation
        self.trg_melspec_fn = instantiate(cfg.preprocessor, highfreq=None, use_grads=True)
        self.generator = instantiate(cfg.generator)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        self.feature_loss = FeatureMatchingLoss()
        self.discriminator_loss = DiscriminatorLoss()
        self.generator_loss = GeneratorLoss()

        self.sample_rate = self._cfg.preprocessor.sample_rate

    def configure_optimizers(self):
        self.optim_g = torch.optim.AdamW(
            self.generator.parameters(), self._cfg.optim.lr, betas=[self._cfg.optim.adam_b1, self._cfg.optim.adam_b2]
        )
        self.optim_d = torch.optim.AdamW(
            itertools.chain(self.msd.parameters(), self.mpd.parameters()),
            self._cfg.optim.lr,
            betas=[self._cfg.optim.adam_b1, self._cfg.optim.adam_b2],
        )

        self.scheduler_g = torch.optim.lr_scheduler.StepLR(
            self.optim_g, step_size=self._cfg.optim.lr_step, gamma=self._cfg.optim.lr_decay,
        )
        self.scheduler_d = torch.optim.lr_scheduler.StepLR(
            self.optim_d, step_size=self._cfg.optim.lr_step, gamma=self._cfg.optim.lr_decay,
        )

        return [self.optim_g, self.optim_d], [self.scheduler_g, self.scheduler_d]

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
        # mel as input for generator
        audio_mel, _ = self.audio_to_melspec_precessor(audio, audio_len)
        # mel as input for L1 mel loss
        audio_trg_mel, _ = self.trg_melspec_fn(audio, audio_len)
        audio = audio.unsqueeze(1)

        audio_pred = self.generator(x=audio_mel)
        audio_pred_mel, _ = self.trg_melspec_fn(audio_pred.squeeze(1), audio_len)

        # train discriminator
        self.optim_d.zero_grad()
        mpd_score_real, mpd_score_gen, _, _ = self.mpd(y=audio, y_hat=audio_pred.detach())
        loss_disc_mpd, _, _ = self.discriminator_loss(
            disc_real_outputs=mpd_score_real, disc_generated_outputs=mpd_score_gen
        )
        msd_score_real, msd_score_gen, _, _ = self.msd(y=audio, y_hat=audio_pred.detach())
        loss_disc_msd, _, _ = self.discriminator_loss(
            disc_real_outputs=msd_score_real, disc_generated_outputs=msd_score_gen
        )
        loss_d = loss_disc_msd + loss_disc_mpd
        self.manual_backward(loss_d, self.optim_d)
        self.optim_d.step()

        # train generator
        self.optim_g.zero_grad()
        loss_mel = F.l1_loss(audio_pred_mel, audio_trg_mel) * 45
        _, mpd_score_gen, fmap_mpd_real, fmap_mpd_gen = self.mpd(y=audio, y_hat=audio_pred)
        _, msd_score_gen, fmap_msd_real, fmap_msd_gen = self.msd(y=audio, y_hat=audio_pred)
        loss_fm_mpd = self.feature_loss(fmap_r=fmap_mpd_real, fmap_g=fmap_mpd_gen)
        loss_fm_msd = self.feature_loss(fmap_r=fmap_msd_real, fmap_g=fmap_msd_gen)
        loss_gen_mpd, _ = self.generator_loss(disc_outputs=mpd_score_gen)
        loss_gen_msd, _ = self.generator_loss(disc_outputs=msd_score_gen)
        loss_g = loss_gen_msd + loss_gen_mpd + loss_fm_msd + loss_fm_mpd + loss_mel
        self.manual_backward(loss_g, self.optim_g)
        self.optim_g.step()

        metrics = {
            "g_l1_loss": loss_mel,
            "g_loss_fm_mpd": loss_fm_mpd,
            "g_loss_fm_msd": loss_fm_msd,
            "g_loss_gen_mpd": loss_gen_mpd,
            "g_loss_gen_msd": loss_gen_msd,
            "g_loss": loss_g,
            "d_loss_mpd": loss_disc_mpd,
            "d_loss_msd": loss_disc_msd,
            "d_loss": loss_d,
            "global_step": self.global_step,
            "lr": self.optim_g.param_groups[0]['lr'],
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        audio, audio_len = batch
        audio_mel, audio_mel_len = self.audio_to_melspec_precessor(audio, audio_len)
        audio_pred = self(spec=audio_mel)

        audio_pred_mel, _ = self.audio_to_melspec_precessor(audio_pred.squeeze(1), audio_len)
        loss_mel = F.l1_loss(audio_mel, audio_pred_mel)

        self.log("val_loss", loss_mel, prog_bar=True, sync_dist=True)

        # plot audio once per epoch
        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            clips = []
            specs = []
            for i in range(min(5, audio.shape[0])):
                clips += [
                    wandb.Audio(
                        audio[i, : audio_len[i]].data.cpu().numpy(),
                        caption=f"real audio {i}",
                        sample_rate=self.sample_rate,
                    ),
                    wandb.Audio(
                        audio_pred[i, 0, : audio_len[i]].data.cpu().numpy().astype('float32'),
                        caption=f"generated audio {i}",
                        sample_rate=self.sample_rate,
                    ),
                ]
                specs += [
                    wandb.Image(
                        plot_spectrogram_to_numpy(audio_mel[i, :, : audio_mel_len[i]].data.cpu().numpy()),
                        caption=f"real audio {i}",
                    ),
                    wandb.Image(
                        plot_spectrogram_to_numpy(audio_pred_mel[i, :, : audio_mel_len[i]].data.cpu().numpy()),
                        caption=f"generated audio {i}",
                    ),
                ]

            self.logger.experiment.log({"audio": clips, "specs": specs}, commit=False)

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
