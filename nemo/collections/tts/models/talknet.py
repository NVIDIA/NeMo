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
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch import nn
from torch.nn import functional as F

from nemo.collections.asr.data.audio_to_text import AudioToCharWithDursF0Dataset
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.modules.talknet import GaussianEmbedding, MaskedInstanceNorm1d, StyleResidual
from nemo.core.classes import ModelPT
from nemo.core.classes.common import typecheck


class TalkNetDursModel(ModelPT):
    """TalkNet's durations prediction pipeline."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)
        typecheck.set_typecheck_enabled(enabled=False)

        cfg = self._cfg
        self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**cfg.train_ds.dataset.vocab)
        self.embed = nn.Embedding(len(self.vocab.labels), cfg.d_char)
        self.model = instantiate(cfg.model)
        d_out = cfg.model.jasper[-1].filters
        self.proj = nn.Conv1d(d_out, 1, kernel_size=1)

    def forward(self, text, text_len):
        x, x_len = self.embed(text).transpose(1, 2), text_len
        y, _ = self.model(x, x_len)
        durs = self.proj(y).squeeze(1)
        return durs

    @staticmethod
    def _metrics(true_durs, true_text_len, pred_durs):
        loss = F.mse_loss(pred_durs, (true_durs + 1).float().log(), reduction='none')
        mask = get_mask_from_lengths(true_text_len)
        loss *= mask.float()
        loss = loss.sum() / mask.sum()

        durs_pred = pred_durs.exp() - 1
        durs_pred[durs_pred < 0.0] = 0.0
        durs_pred = durs_pred.round().long()
        acc = ((true_durs == durs_pred) * mask).sum().float() / mask.sum() * 100

        return loss, acc

    def training_step(self, batch, batch_idx):
        _, _, text, text_len, durs, *_ = batch
        pred_durs = self(text=text, text_len=text_len)
        loss, acc = self._metrics(true_durs=durs, true_text_len=text_len, pred_durs=pred_durs,)
        train_log = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'progress_bar': train_log, 'log': train_log}

    def validation_step(self, batch, batch_idx):
        _, _, text, text_len, durs, *_ = batch
        pred_durs = self(text=text, text_len=text_len)
        loss, acc = self._metrics(true_durs=durs, true_text_len=text_len, pred_durs=pred_durs,)
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = torch.stack([x['acc'] for x in outputs]).mean()
        val_log = {'val_loss': loss, 'val_acc': acc}
        return {'val_loss': loss, 'log': val_log}

    @staticmethod
    def _loader(cfg):
        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(  # noqa
            dataset=dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params,
        )

    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    @classmethod
    def list_available_models(cls):
        """Empty."""
        pass


class TalkNetPitchModel(ModelPT):
    """TalkNet's pitch prediction pipeline."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)
        typecheck.set_typecheck_enabled(enabled=False)

        cfg = self._cfg
        self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**cfg.train_ds.dataset.vocab)
        self.embed = GaussianEmbedding(self.vocab, cfg.d_char)
        self.model = instantiate(cfg.model)
        d_out = cfg.model.jasper[-1].filters
        self.sil_proj = nn.Conv1d(d_out, 1, kernel_size=1)
        self.body_proj = nn.Conv1d(d_out, 1, kernel_size=1)
        self.f0_mean, self.f0_std = cfg.f0_mean, cfg.f0_std

    def forward(self, text, text_len, durs):
        x, x_len = self.embed(text, durs).transpose(1, 2), durs.sum(-1)
        y, _ = self.model(x, x_len)
        f0_sil = self.sil_proj(y).squeeze(1)
        f0_body = self.body_proj(y).squeeze(1)
        return f0_sil, f0_body

    def _metrics(self, true_f0, true_f0_mask, pred_f0_sil, pred_f0_body):
        sil_mask = true_f0 < 1e-5
        sil_gt = sil_mask.long()
        sil_loss = F.binary_cross_entropy_with_logits(input=pred_f0_sil, target=sil_gt.float(), reduction='none',)
        sil_loss *= true_f0_mask.type_as(sil_loss)
        sil_loss = sil_loss.sum() / true_f0_mask.sum()
        sil_acc = ((torch.sigmoid(pred_f0_sil) > 0.5).long() == sil_gt).float()  # noqa
        sil_acc *= true_f0_mask.type_as(sil_acc)
        sil_acc = sil_acc.sum() / true_f0_mask.sum()

        body_mse = F.mse_loss(pred_f0_body, (true_f0 - self.f0_mean) / self.f0_std, reduction='none')
        body_mask = ~sil_mask
        body_mse *= body_mask.type_as(body_mse)  # noqa
        body_mse = body_mse.sum() / body_mask.sum()  # noqa
        body_mae = ((pred_f0_body * self.f0_std + self.f0_mean) - true_f0).abs()
        body_mae *= body_mask.type_as(body_mae)  # noqa
        body_mae = body_mae.sum() / body_mask.sum()  # noqa

        loss = sil_loss + body_mse

        return loss, sil_acc, body_mae

    def training_step(self, batch, batch_idx):
        _, audio_len, text, text_len, durs, f0, f0_mask = batch
        pred_f0_sil, pred_f0_body = self(text=text, text_len=text_len, durs=durs)
        loss, sil_acc, body_mae = self._metrics(
            true_f0=f0, true_f0_mask=f0_mask, pred_f0_sil=pred_f0_sil, pred_f0_body=pred_f0_body,
        )
        train_log = {'train_loss': loss, 'train_sil_acc': sil_acc, 'train_body_mae': body_mae}
        return {'loss': loss, 'progress_bar': train_log, 'log': train_log}

    def validation_step(self, batch, batch_idx):
        _, _, text, text_len, durs, f0, f0_mask = batch
        pred_f0_sil, pred_f0_body = self(text=text, text_len=text_len, durs=durs)
        loss, sil_acc, body_mae = self._metrics(
            true_f0=f0, true_f0_mask=f0_mask, pred_f0_sil=pred_f0_sil, pred_f0_body=pred_f0_body,
        )
        return {'loss': loss, 'sil_acc': sil_acc, 'body_mae': body_mae}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        sil_acc = torch.stack([x['sil_acc'] for x in outputs]).mean()
        body_mae = torch.stack([x['body_mae'] for x in outputs]).mean()
        val_log = {'val_loss': loss, 'val_sil_acc': sil_acc, 'val_body_mae': body_mae}
        return {'val_loss': loss, 'log': val_log}

    @staticmethod
    def _loader(cfg):
        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(  # noqa
            dataset=dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params,
        )

    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    @classmethod
    def list_available_models(cls):
        """Empty."""
        pass


class TalkNetSpectModel(SpectrogramGenerator):
    """TalkNet's mel spectrogram prediction pipeline."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)
        typecheck.set_typecheck_enabled(enabled=False)

        cfg = self._cfg
        self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**cfg.train_ds.dataset.vocab)
        self.preprocessor = instantiate(cfg.preprocessor)
        self.embed = GaussianEmbedding(self.vocab, cfg.d_char)
        self.norm_f0 = MaskedInstanceNorm1d(1)
        self.res_f0 = StyleResidual(cfg.d_char, 1, kernel_size=3)
        self.model = instantiate(cfg.model)
        d_out = cfg.model.jasper[-1].filters
        self.proj = nn.Conv1d(d_out, cfg.n_mels, kernel_size=1)

    def forward(self, text, text_len, durs, f0):
        x, x_len = self.embed(text, durs).transpose(1, 2), durs.sum(-1)
        f0, f0_mask = f0.clone(), f0 > 0.0
        f0 = self.norm_f0(f0.unsqueeze(1), f0_mask)
        f0[~f0_mask.unsqueeze(1)] = 0.0
        x = self.res_f0(x, f0)
        y, _ = self.model(x, x_len)
        mel = self.proj(y)
        return mel

    @staticmethod
    def _metrics(true_mel, true_mel_len, pred_mel):
        loss = F.mse_loss(pred_mel, true_mel, reduction='none').mean(dim=-2)
        mask = get_mask_from_lengths(true_mel_len)
        loss *= mask.float()
        loss = loss.sum() / mask.sum()
        return loss

    def training_step(self, batch, batch_idx):
        audio, audio_len, text, text_len, durs, f0, f0_mask = batch
        mel, mel_len = self.preprocessor(audio, audio_len)
        pred_mel = self(text=text, text_len=text_len, durs=durs, f0=f0)
        loss = self._metrics(true_mel=mel, true_mel_len=mel_len, pred_mel=pred_mel)
        train_log = {'train_loss': loss}
        return {'loss': loss, 'progress_bar': train_log, 'log': train_log}

    def validation_step(self, batch, batch_idx):
        audio, audio_len, text, text_len, durs, f0, f0_mask = batch
        mel, mel_len = self.preprocessor(audio, audio_len)
        pred_mel = self(text=text, text_len=text_len, durs=durs, f0=f0)
        loss = self._metrics(true_mel=mel, true_mel_len=mel_len, pred_mel=pred_mel)
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        val_log = {'val_loss': loss}
        return {'val_loss': loss, 'log': val_log}

    @staticmethod
    def _loader(cfg):
        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(  # noqa
            dataset=dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params,
        )

    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    @classmethod
    def list_available_models(cls):
        """Empty."""
        pass

    def parse(self, text: str, **kwargs) -> torch.Tensor:
        return torch.tensor(self.vocab.encode(text)).long().unsqueeze(0).to(self.device)

    def generate_spectrogram(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        assert hasattr(self, '_durs_model') and hasattr(self, '_pitch_model')

        # Durs
        text = [
            AudioToCharWithDursF0Dataset.interleave(
                x=torch.empty(len(t) + 1, dtype=torch.long, device=t.device).fill_(self.vocab.blank), y=t,
            )
            for t in tokens
        ]
        text = AudioToCharWithDursF0Dataset.merge(text, value=self.vocab.pad, dtype=torch.long)
        text_len = torch.tensor(text.shape[-1], dtype=torch.long).unsqueeze(0)
        durs = self._durs_model(text, text_len)
        durs = durs.exp() - 1
        durs[durs < 0.0] = 0.0
        durs = durs.round().long()

        # Pitch
        f0_sil, f0_body = self._pitch_model(text, text_len, durs)
        sil_mask = f0_sil.sigmoid() > 0.5
        f0 = f0_body * self._pitch_model.f0_std + self._pitch_model.f0_mean
        f0 = (~sil_mask * f0).float()

        # Spect
        mel = self(text, text_len, durs, f0)

        return mel
