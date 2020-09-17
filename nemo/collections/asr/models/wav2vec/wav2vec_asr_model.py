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
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, cast, List

import torch
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.wav2vec.wav2vec_base import Wav2VecBase
from nemo.collections.asr.models.wav2vec.wav2vec_config import Wav2VecCTCEncoderConfig
from nemo.collections.asr.models.wav2vec.wav2vec_model import Wav2VecEncoderModel
from nemo.core.classes.common import typecheck, PretrainedModelInfo
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn


class Wav2VecCTCEncoder(nn.Module):
    def __init__(self,
                 wav2vec_encoder: Wav2VecEncoderModel,
                 cfg: Wav2VecCTCEncoderConfig,
                 encoder_dim: int,
                 trainer: Trainer):
        super().__init__()
        self.trainer = trainer
        self.final_dropout = nn.Dropout(cfg.final_dropout)
        # Add 1 for blank char
        self.vocabulary = cfg.vocabulary
        self.freeze_encoder_after_steps = cfg.freeze_encoder_after_steps
        self._num_classes = len(self.vocabulary) + 1
        self.apply_mask = cfg.mask.apply_mask
        self.wav2vec_encoder = wav2vec_encoder

        if self.apply_mask:
            # Override encoder mask cfg with ctc encoder mask cfg
            self.wav2vec_encoder.mask_cfg = cfg.mask
        self.wav2vec_encoder.remove_pretraining_modules()

        self.proj = self.linear(encoder_dim, self._num_classes)

    def linear(self, in_features, out_features, bias=True):
        m = nn.Linear(in_features, out_features, bias)
        nn.init.xavier_uniform_(m.weight)
        if bias:
            nn.init.constant_(m.bias, 0.0)
        return m

    def forward(self, audio_signal, padding_mask):
        freeze_encoder_at_step = self.freeze_encoder_after_steps is not None and \
                                 self.freeze_encoder_after_steps <= self.trainer.global_step

        if freeze_encoder_at_step:
            with torch.no_grad():
                x, padding_mask = self.wav2vec_encoder.extract_features(
                    source=audio_signal,
                    padding_mask=padding_mask,
                    mask=self.apply_mask and self.training
                )
        else:
            x, padding_mask = self.wav2vec_encoder.extract_features(
                source=audio_signal,
                padding_mask=padding_mask,
                mask=self.apply_mask and self.training
            )

        x = self.final_dropout(x)
        x = self.proj(x)

        non_padding_mask = ~padding_mask
        output_lengths = non_padding_mask.long().sum(-1)
        return x, output_lengths

    @property
    def num_classes_with_blank(self):
        return self._num_classes


class Wav2VecASRModel(Wav2VecBase, ASRModel):
    def transcribe(self, paths2audio_files: List[str], batch_size: int = 4) -> List[str]:
        pass

    def __init__(self, encoder: Wav2VecEncoderModel, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(Wav2VecCTCEncoderConfig)
        cfg = cfg.get('params', {})
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg = OmegaConf.merge(schema, cfg)
        cfg = cast(Wav2VecCTCEncoderConfig, cfg)

        self.encoder = Wav2VecCTCEncoder(
            wav2vec_encoder=encoder,
            cfg=cfg,
            encoder_dim=encoder.final_dim,
            trainer=trainer
        )

        self.loss = CTCLoss(
            num_classes=self.encoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )

        # Setup metric objects
        self._wer = WER(
            vocabulary=self.encoder.vocabulary,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    @typecheck()
    def forward(self, input_signal, padding_mask):
        x, encoded_len = self.encoder(audio_signal=input_signal, padding_mask=padding_mask)
        log_probs = x.log_softmax(-1)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        return log_probs, encoded_len, greedy_predictions

    def model_forward_and_loss(self, batch):
        audio_signal, audio_lengths, transcript, transcript_len, padding_mask = batch
        log_probs, encoded_len, predictions = self.forward(
            input_signal=audio_signal,
            padding_mask=padding_mask
        )

        loss = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        return loss, predictions, transcript, transcript_len

    # PTL-specific methods
    def training_step(self, batch, batch_idx):
        loss, predictions, transcript, transcript_len = self.model_forward_and_loss(batch)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        if (batch_idx + 1) % log_every_n_steps == 0:
            self._wer.update(predictions, transcript, transcript_len)
            wer, _, _ = self._wer.compute()
            self.log('training_batch_wer', wer, prog_bar=True)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('loss', loss, prog_bar=False, on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, predictions, transcript, transcript_len = self.model_forward_and_loss(batch)
        self._wer.update(predictions, transcript, transcript_len)
        wer, wer_num, wer_denom = self._wer.compute()
        self.log_dict({
            'val_loss': loss,
            'val_wer': wer,
        }, sync_dist=True, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, predictions, transcript, transcript_len = self.model_forward_and_loss(batch)
        self._wer.update(predictions, transcript, transcript_len)
        wer, wer_num, wer_denom = self._wer.compute()
        self.log_dict({
            'test_loss': loss,
            'test_wer': wer,
        }, sync_dist=True, prog_bar=True, on_epoch=True)
