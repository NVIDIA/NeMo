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
import contextlib
from dataclasses import dataclass
from typing import Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger

from nemo.collections.common.parts.preprocessing import parsers
from nemo.collections.tts.helpers.helpers import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.collections.tts.losses.fastpitchloss import DurationLoss, MelLoss, PitchLoss
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.modules.fastpitch import FastPitchModule
from nemo.collections.tts.torch.tts_data_types import SpeakerID
from nemo.core.classes import Exportable, ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    Index,
    LengthsType,
    MelSpectrogramType,
    ProbsType,
    RegressionValuesType,
    TokenDurationType,
    TokenIndex,
    TokenLogDurationType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging, model_utils


def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


class FastPitchModel_SSL(ModelPT):
    """FastPitch model (https://arxiv.org/abs/2006.06873) that is used to generate mel spectrogram from text."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        self.learn_alignment = False

        # Setup vocabulary (=tokenizer) and input_fft_kwargs (supported only with self.learn_alignment=True)
        input_fft_kwargs = {}

        self._parser = None
        self._tb_logger = None
        super().__init__(cfg=cfg, trainer=trainer)

        self.bin_loss_warmup_epochs = cfg.get("bin_loss_warmup_epochs", 100)
        self.log_train_images = False

        loss_scale = 0.1 if self.learn_alignment else 1.0
        dur_loss_scale = loss_scale
        pitch_loss_scale = loss_scale
        if "dur_loss_scale" in cfg:
            dur_loss_scale = cfg.dur_loss_scale
        if "pitch_loss_scale" in cfg:
            pitch_loss_scale = cfg.pitch_loss_scale

        self.mel_loss = MelLoss()
        self.pitch_loss = PitchLoss(loss_scale=pitch_loss_scale)
        self.duration_loss = DurationLoss(loss_scale=dur_loss_scale)

        self.aligner = None

        self.preprocessor = instantiate(self._cfg.preprocessor)
        input_fft = None
        output_fft = instantiate(self._cfg.output_fft)
        duration_predictor = instantiate(self._cfg.duration_predictor)
        pitch_predictor = instantiate(self._cfg.pitch_predictor)

        self.content_projection_layer = torch.nn.Linear(self._cfg.content_emb_indim, self._cfg.content_emb_outdim)
        self.speaker_projection_layer = torch.nn.Linear(self._cfg.speaker_emb_indim, self._cfg.speaker_emb_outdim)

        self.fastpitch = FastPitchModule(
            input_fft,
            output_fft,
            duration_predictor,
            pitch_predictor,
            self.aligner,
            cfg.n_speakers,
            cfg.symbols_embedding_dim,
            cfg.pitch_embedding_kernel_size,
            cfg.n_mel_channels,
        )

    @property
    def tb_logger(self):
        if self._tb_logger is None:
            if self.logger is None and self.logger.experiment is None:
                return None
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break
            self._tb_logger = tb_logger
        return self._tb_logger

    def forward(
        self,
        *,
        text,
        durs=None,
        pitch=None,
        speaker=None,
        pace=1.0,
        spec=None,
        attn_prior=None,
        mel_lens=None,
        input_lens=None,
        enc_out=None,
        enc_mask=None,
    ):
        return self.fastpitch(
            text=text,
            durs=durs,
            pitch=pitch,
            speaker=speaker,
            pace=pace,
            spec=spec,
            attn_prior=attn_prior,
            mel_lens=mel_lens,
            input_lens=input_lens,
            enc_out=enc_out,
            enc_mask=enc_mask,
        )

    @typecheck(output_types={"spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType())})
    def generate_spectrogram(
        self, tokens: 'torch.tensor', speaker: Optional[int] = None, pace: float = 1.0
    ) -> torch.tensor:
        if self.training:
            logging.warning("generate_spectrogram() is meant to be called in eval mode.")
        if isinstance(speaker, int):
            speaker = torch.tensor([speaker]).to(self.device)
        spect, *_ = self(text=tokens, durs=None, pitch=None, speaker=speaker, pace=pace)
        return spect

    def compute_encoding(self, content_embedding, speaker_embedding):
        # content embedding is (B, C, T)
        # speaker embedding is (B, C)
        # pitch_contour is (B, T)
        content_embedding = content_embedding.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
        content_embedding_projected = self.content_projection_layer(content_embedding)
        content_embedding_projected = content_embedding_projected.permute(0, 2, 1)
        speaker_embedding_projected = self.speaker_projection_layer(speaker_embedding)
        speaker_embedding_repeated = speaker_embedding_projected[:, :, None].repeat(
            1, 1, content_embedding_projected.shape[2]
        )

        encoded = torch.cat([content_embedding_projected, speaker_embedding_repeated], dim=1)

        encoded = encoded.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

        return encoded

    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        audio_lens = batch["audio_len"]
        content_embedding = batch["content_embedding"]
        encoded_len = batch["encoded_len"]
        speaker_embedding = batch["speaker_embedding"]
        mels = batch["mel_spectrogram"]
        spec_len = batch["mel_len"]
        pitch = batch["pitch_contour"]

        # mels, spec_len = self.preprocessor(input_signal=audio, length=audio_lens)

        # print("mels.shape: ", mels.shape)
        # print("mels_dl.shape: ", mels_dl.shape)
        # print("mels_lens_dl: ", mels_lens_dl)
        # print("spec_len: ", spec_len)
        enc_out = self.compute_encoding(content_embedding, speaker_embedding)
        enc_mask = mask_from_lens(encoded_len)
        durs = torch.ones_like(enc_mask) * 4.0
        enc_mask = enc_mask[:, :, None]
        print("enc_out.shape: ", enc_out.shape)
        print("enc_mask.shape: ", enc_mask.shape)
        print("durs.shape: ", durs.shape)

        mels_pred, _, _, log_durs_pred, pitch_pred, attn_soft, attn_logprob, attn_hard, attn_hard_dur, pitch = self(
            text=None,
            durs=durs,
            pitch=pitch,
            speaker=None,
            pace=1.0,
            spec=mels if self.learn_alignment else None,
            attn_prior=None,
            mel_lens=spec_len,
            input_lens=None,
            enc_out=enc_out,
            enc_mask=enc_mask,
        )

        if durs is None:
            durs = attn_hard_dur

        mel_loss = self.mel_loss(spect_predicted=mels_pred, spect_tgt=mels)
        dur_loss = self.duration_loss(log_durs_predicted=log_durs_pred, durs_tgt=durs, len=encoded_len)
        loss = mel_loss + dur_loss

        pitch_loss = self.pitch_loss(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=encoded_len)
        loss += pitch_loss

        self.log("t_loss", loss)
        self.log("t_mel_loss", mel_loss)
        self.log("t_dur_loss", dur_loss)
        self.log("t_pitch_loss", pitch_loss)

        # Log images to tensorboard
        if self.log_train_images and isinstance(self.logger, TensorBoardLogger):
            self.log_train_images = False

            self.tb_logger.add_image(
                "train_mel_target",
                plot_spectrogram_to_numpy(mels[0].data.cpu().float().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            spec_predict = mels_pred[0].data.cpu().float().numpy()
            self.tb_logger.add_image(
                "train_mel_predicted", plot_spectrogram_to_numpy(spec_predict), self.global_step, dataformats="HWC",
            )

        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch["audio"]
        audio_lens = batch["audio_len"]
        content_embedding = batch["content_embedding"]
        encoded_len = batch["encoded_len"]
        speaker_embedding = batch["speaker_embedding"]
        mels = batch["mel_spectrogram"]
        spec_len = batch["mel_len"]
        pitch = batch["pitch_contour"]

        enc_out = self.compute_encoding(content_embedding, speaker_embedding)
        enc_mask = mask_from_lens(encoded_len)
        durs = torch.ones_like(enc_mask) * 4.0
        enc_mask = enc_mask[:, :, None]
        print("enc_out.shape: ", enc_out.shape)
        print("enc_mask.shape: ", enc_mask.shape)
        print("durs.shape: ", durs.shape)

        # # Calculate val loss on ground truth durations to better align L2 loss in time
        mels_pred, _, _, log_durs_pred, pitch_pred, _, _, _, attn_hard_dur, pitch = self(
            text=None,
            durs=durs,
            pitch=pitch,
            speaker=None,
            pace=1.0,
            spec=None,
            attn_prior=None,
            mel_lens=spec_len,
            input_lens=None,
            enc_out=enc_out,
            enc_mask=enc_mask,
        )
        # if durs is None:
        #     durs = attn_hard_dur

        mel_loss = self.mel_loss(spect_predicted=mels_pred, spect_tgt=mels)
        dur_loss = self.duration_loss(log_durs_predicted=log_durs_pred, durs_tgt=durs, len=encoded_len)
        pitch_loss = self.pitch_loss(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=encoded_len)
        loss = mel_loss + dur_loss + pitch_loss

        return {
            "val_loss": loss,
            "mel_loss": mel_loss,
            "dur_loss": dur_loss,
            "pitch_loss": pitch_loss,
            "mel_target": mels if batch_idx == 0 else None,
            "mel_pred": mels_pred if batch_idx == 0 else None,
        }

    def validation_epoch_end(self, outputs):
        collect = lambda key: torch.stack([x[key] for x in outputs]).mean()
        val_loss = collect("val_loss")
        mel_loss = collect("mel_loss")
        dur_loss = collect("dur_loss")
        pitch_loss = collect("pitch_loss")
        self.log("v_loss", val_loss)
        self.log("v_mel_loss", mel_loss)
        self.log("v_dur_loss", dur_loss)
        self.log("v_pitch_loss", pitch_loss)

        _, _, _, _, spec_target, spec_predict = outputs[0].values()

        if isinstance(self.logger, TensorBoardLogger):
            self.tb_logger.add_image(
                "val_mel_target",
                plot_spectrogram_to_numpy(spec_target[0].data.cpu().float().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            spec_predict = spec_predict[0].data.cpu().float().numpy()
            self.tb_logger.add_image(
                "val_mel_predicted", plot_spectrogram_to_numpy(spec_predict), self.global_step, dataformats="HWC",
            )
            self.log_train_images = True

    def __setup_dataloader_from_config(self, cfg):
        dataset = instantiate(cfg.dataset)

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.pad_collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        return []
