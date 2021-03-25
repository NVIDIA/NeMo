# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger

from nemo.collections.asr.data.audio_to_text import AudioToCharWithDursF0Dataset
from nemo.collections.tts.helpers.helpers import plot_spectrogram_to_numpy
from nemo.collections.tts.losses.fastspeech2loss import L2MelLoss
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.modules.fastspeech2 import FastSpeech2Encoder, MelSpecDecoder, VarianceAdaptor
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.utils import logging


@dataclass
class PreprocessorParams:
    pad_value: float = MISSING


@dataclass
class Preprocessor:
    cls: str = MISSING
    params: PreprocessorParams = PreprocessorParams()


@dataclass
class FastSpeech2Config:
    fastspeech2: Dict[Any, Any] = MISSING
    preprocessor: Preprocessor = Preprocessor()
    # TODO: may need more params
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None


class DurationLoss(torch.nn.Module):
    def forward(self, log_duration_pred, duration_target, mask):
        log_duration_target = torch.log(duration_target + 1)
        return torch.nn.functional.mse_loss(log_duration_pred, log_duration_target)


class FastSpeech2Model(SpectrogramGenerator):
    """FastSpeech 2 model used to convert between text (phonemes) and mel-spectrograms."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        self.vocab = None
        if cfg.train_ds.dataset._target_ == "nemo.collections.asr.data.audio_to_text.AudioToCharWithDursF0Dataset":
            self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**cfg.train_ds.dataset.vocab)

        schema = OmegaConf.structured(FastSpeech2Config)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        # Ensure passed cfg is compliant with schema
        OmegaConf.merge(cfg, schema)

        self.energy = cfg.add_energy_predictor
        self.pitch = cfg.add_pitch_predictor
        self.duration = True
        self.duration_coeff = cfg.duration_coeff

        self.audio_to_melspec_precessor = instantiate(self._cfg.preprocessor)
        self.encoder = FastSpeech2Encoder()
        self.variance_adapter = VarianceAdaptor(pitch=self.pitch, energy=self.energy)
        self.mel_decoder = MelSpecDecoder()
        self.loss = L2MelLoss()
        self.mseloss = torch.nn.MSELoss()
        self.durationloss = DurationLoss()

        self.log_train_images = False

    # @property
    # def input_types(self):
    #     return {"text": NeuralType(('B', 'T'), TokenIndex()), "text_lengths": NeuralType(('B'), LengthsType())}

    # @property
    # def output_types(self):
    #     # May need to condition on OperationMode.training vs OperationMode.validation
    #     pass

    @typecheck()
    def forward(self, *, spec_len, text, text_length, durations=None, pitch=None, energies=None):
        with typecheck.disable_checks():
            encoded_text, encoded_text_mask = self.encoder(text=text, text_lengths=text_length)
            aligned_text, log_dur_preds, pitch_preds, energy_preds, spec_len = self.variance_adapter(
                x=encoded_text,
                x_len=text_length,
                dur_target=durations,
                pitch_target=pitch,
                energy_target=energies,
                spec_len=spec_len
            )
            mel = self.mel_decoder(decoder_input=aligned_text, lengths=spec_len)
            return mel, log_dur_preds, pitch_preds, energy_preds, encoded_text_mask

    def training_step(self, batch, batch_idx):
        if self.vocab is None:
            f, fl, t, tl, durations, pitch, energies = batch
        else:
            pitch, energies = None, None
            f, fl, t, tl, durations, pitch, _ = batch
        spec, spec_len = self.audio_to_melspec_precessor(f, fl)
        mel, log_dur_preds, pitch_preds, energy_preds, encoded_text_mask = self(
            spec_len=spec_len, text=t, text_length=tl, durations=durations, pitch=pitch, energies=energies
        )
        total_loss = self.loss(spec_pred=mel, spec_target=spec, spec_target_len=spec_len, pad_value=-11.52)
        self.log(name="train_mel_loss", value=total_loss.clone().detach())
        if self.duration:
            dur_loss = self.durationloss(log_dur_preds, durations.float(), encoded_text_mask)
            dur_loss *= self.duration_coeff
            self.log(name="train_dur_loss", value=dur_loss)
            total_loss += dur_loss
        if self.pitch:
            pitch_loss = self.mseloss(pitch_preds, pitch)
            total_loss += pitch_loss
            self.log(name="train_pitch_loss", value=pitch_loss)
        if self.energy:
            energy_loss = self.mseloss(energy_preds, energies)
            total_loss += energy_loss
            self.log(name="train_energy_loss", value=energy_loss)
        self.log(name="train_loss", value=total_loss)
        # if (self.global_step + 1) % 200 == 0:
        #     logging.info(f"train_loss: {total_loss}")
        #     logging.info(f"train_mel_loss: {loss}")
        #     if self.duration:
        #         logging.info(f"train_dur_loss: {dur_loss}")
        #     if self.pitch:
        #         logging.info(f"train_pitch_loss: {pitch_loss}")
        #     if self.energy:
        #         logging.info(f"train_energy_loss: {energy_loss}")
        return {"loss": total_loss, "outputs": [spec, mel]}

    def training_epoch_end(self, training_step_outputs):
        if self.log_train_images and self.logger is not None and self.logger.experiment is not None:
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break
            spec_target, spec_predict = training_step_outputs[0]["outputs"]
            tb_logger.add_image(
                "train_mel_target",
                plot_spectrogram_to_numpy(spec_target[0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            spec_predict = spec_predict[0].data.cpu().numpy()
            tb_logger.add_image(
                "train_mel_predicted", plot_spectrogram_to_numpy(spec_predict.T), self.global_step, dataformats="HWC",
            )
            self.log_train_images = False

    def validation_step(self, batch, batch_idx):
        if self.vocab is None:
            f, fl, t, tl, _ = batch
        else:
            f, fl, t, tl, _, _, _ = batch
        spec, spec_len = self.audio_to_melspec_precessor(f, fl)
        mel, log_dur_preds, _, _, _ = self(spec_len=spec_len, text=t, text_length=tl)
        loss = self.loss(spec_pred=mel, spec_target=spec, spec_target_len=spec_len, pad_value=-11.52)
        return {
            "val_loss": loss,
            "mel_target": spec,
            "mel_pred": mel,
        }

    def validation_epoch_end(self, outputs):
        if self.logger is not None and self.logger.experiment is not None:
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break
            _, spec_target, spec_predict = outputs[0].values()
            tb_logger.add_image(
                "val_mel_target",
                plot_spectrogram_to_numpy(spec_target[0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            spec_predict = spec_predict[0].data.cpu().numpy()
            tb_logger.add_image(
                "val_mel_predicted", plot_spectrogram_to_numpy(spec_predict.T), self.global_step, dataformats="HWC",
            )
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()  # This reduces across batches, not workers!
        self.log('val_loss', avg_loss, sync_dist=True)

        self.log_train_images = True

    # @typecheck(
    #     input_types={"text": NeuralType(('B', 'T'), TokenIndex()), "text_lengths": NeuralType(('B'), LengthsType())},
    #     output_types={"spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType())},
    # )
    def generate_spectrogram(self, tokens: torch.Tensor) -> torch.Tensor:
        # TODO
        pass

    def parse(self, str_input: str) -> torch.Tensor:
        # TODO
        pass

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
                with open_dict(cfg.dataloader_params):
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
