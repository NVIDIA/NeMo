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
from torch import nn
from torch.nn import functional as F

from nemo.collections.asr.data.audio_to_text import AudioToCharWithDursF0Dataset
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.core.classes import Loss, ModelPT
from nemo.core.classes.common import typecheck
from nemo.core.neural_types.elements import (
    EmbeddedTextType,
    LengthsType,
    LossType,
    MelSpectrogramType,
    SpectrogramType,
    FloatType,
)
from nemo.core.neural_types.neural_type import NeuralType


class GaussianEmbedding(nn.Module):
    """Gaussian embedding layer.."""

    EPS = 1e-6

    def __init__(
        self,
        vocab,
        d_emb,
        sigma_c=2.0,
        merge_blanks=False,
    ):
        super().__init__()

        self.embed = nn.Embedding(len(vocab.labels), d_emb)
        self.pad = vocab.pad
        self.sigma_c = sigma_c
        self.merge_blanks = merge_blanks

    def forward(self, text, durs):
        """See base class."""
        # Fake padding
        text = F.pad(text, [0, 2, 0, 0], value=self.pad)
        durs = F.pad(durs, [0, 2, 0, 0], value=0)

        repeats = AudioToCharWithDursF0Dataset.repeat_merge(text, durs, self.pad)
        total_time = repeats.shape[-1]

        # Centroids: [B,T,N]
        c = (durs / 2.0) + F.pad(torch.cumsum(durs, dim=-1)[:, :-1], [1, 0, 0, 0], value=0)
        c = c.unsqueeze(1).repeat(1, total_time, 1)

        # Sigmas: [B,T,N]
        sigmas = durs
        sigmas = sigmas.float() / self.sigma_c
        sigmas = sigmas.unsqueeze(1).repeat(1, total_time, 1) + self.EPS
        assert c.shape == sigmas.shape

        # Times at indexes
        t = (
            torch.arange(total_time, device=c.device)
            .view(1, -1, 1)
            .repeat(durs.shape[0], 1, durs.shape[-1])
            .float()
        )
        t = t + 0.5

        ns = slice(None)
        if self.merge_blanks:
            ns = slice(1, None, 2)

        # Weights: [B,T,N]
        d = torch.distributions.normal.Normal(c, sigmas)
        w = d.log_prob(t).exp()[:, :, ns]  # [B,T,N]
        pad_mask = (text == self.pad)[:, ns].unsqueeze(1).repeat(1, total_time, 1)
        w.masked_fill_(pad_mask, 0.0)  # noqa
        w = w / (w.sum(-1, keepdim=True) + self.EPS)
        pad_mask = (repeats == self.pad).unsqueeze(-1).repeat(1, 1, text[:, ns].size(1))  # noqa
        w.masked_fill_(pad_mask, 0.0)  # noqa
        pad_mask[:, :, :-1] = False
        w.masked_fill_(pad_mask, 1.0)  # noqa

        # Embeds
        u = torch.bmm(w, self.embed(text)[:, ns, :])  # [B,T,E]

        return u


class TalkNetDursLoss(Loss):
    """A Loss module that computes loss for TalkNet durations predictor."""

    @property
    def input_types(self):
        return {
            'durs_true': NeuralType(('B', 'T'), LengthsType()),
            'len_true': NeuralType(('B',), LengthsType()),
            'durs_pred': NeuralType(('B', 'T'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            'loss': NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, *, durs_true, len_true, durs_pred):
        loss = F.mse_loss(durs_pred, (durs_true + 1).float().log(), reduction='none')
        mask = get_mask_from_lengths(len_true)
        loss *= mask.float()
        return loss.sum() / mask.sum()


class TalkNetDursModel(ModelPT):
    """TalkNet's durations prediction pipeline."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):  # noqa
        super().__init__(cfg=cfg, trainer=trainer)

        cfg = self._cfg
        self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**cfg.train_ds.dataset.vocab)
        self.emb = nn.Embedding(len(self.vocab.labels), cfg.d_char)
        self.model = instantiate(cfg.model)
        self.rz = nn.Conv1d(cfg.model.jasper[-1].filters, 1, kernel_size=1)
        self.loss = TalkNetDursLoss()

    @property
    def input_types(self):
        return {
            'text': NeuralType(('B', 'T'), EmbeddedTextType()),
            'text_len': NeuralType(('B',), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            'durs': NeuralType(('B', 'T'), LengthsType()),
        }

    @typecheck()
    def forward(self, *, text, text_len):
        x, x_len = self.emb(text).transpose(1, 2), text_len
        y, _ = self.model(audio_signal=x, length=x_len)
        y = self.rz(y)
        durs = y.squeeze(1)  # Contracting last 1-dim.
        return durs

    @staticmethod
    def _acc(durs_true, len_true, durs_pred):
        mask = get_mask_from_lengths(len_true)
        durs_pred = durs_pred.exp() - 1
        durs_pred[durs_pred < 0.0] = 0.0
        durs_pred = durs_pred.round().long()
        return ((durs_true == durs_pred) * mask).sum().float() / mask.sum() * 100

    def training_step(self, batch, batch_idx):
        _, _, text, text_len, durs = batch
        durs_pred = self(text=text, text_len=text_len)
        loss = self.loss(durs_true=durs, len_true=text_len, durs_pred=durs_pred)  # noqa
        acc = self._acc(durs_true=durs, len_true=text_len, durs_pred=durs_pred)

        return {
            'loss': loss,
            'progress_bar': {'train_loss': loss, 'train_acc': acc},
            'log': {'train_loss': loss, 'train_acc': acc},
        }

    def validation_step(self, batch, batch_idx):
        _, _, text, text_len, durs = batch
        durs_pred = self(text=text, text_len=text_len)
        loss = self.loss(durs_true=durs, len_true=text_len, durs_pred=durs_pred)  # noqa
        acc = self._acc(durs_true=durs, len_true=text_len, durs_pred=durs_pred)
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        return {
            'val_loss': avg_loss,
            'log': {'val_loss': avg_loss, 'val_acc': avg_acc},
        }

    @staticmethod
    def _loader(cfg):
        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(  # noqa
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            **cfg.dataloader_params,
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

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)

        cfg = self._cfg
        self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**cfg.train_ds.dataset.vocab)
        self.emb = GaussianEmbedding(self.vocab, cfg.d_char)
        self.model = instantiate(cfg.model)
        d_out = cfg.model.jasper[-1].filters
        self.sil_proj = nn.Conv1d(d_out, 1, kernel_size=1)
        self.body_proj = nn.Conv1d(d_out, 1, kernel_size=1)

    def forward(self, text, text_len, durs):
        x, x_len = self.emb(text, durs).transpose(1, 2), text_len
        y, _ = self.model(x, x_len)
        f0_sil = self.sil_proj(y).squeeze(1)
        f0_body = self.body_rz(y).squeeze(1)
        return f0_sil, f0_body

    F0_MEAN, F0_STD = 150.23434143088116, 42.795667026124704

    def _metrics(self, true_mel_mask, true_f0, pred_f0_sil, pred_f0_body):
        sil_mask = true_f0 < 1e-5
        sil_gt = sil_mask.long()
        sil_loss = F.binary_cross_entropy_with_logits(
            input=pred_f0_sil,
            target=sil_gt.float(),
            reduction='none',
        )
        sil_loss *= true_mel_mask.type_as(sil_loss)
        sil_loss = sil_loss.sum() / true_mel_mask.sum()
        sil_acc = ((torch.sigmoid(output.sil) > 0.5).long() == sil_gt).float()  # noqa
        sil_acc *= true_mel_mask.type_as(sil_acc)
        sil_acc = sil_acc.sum() / true_mel_mask.sum()

        body_mse = F.mse_loss(pred_f0_body, (true_f0 - self.F0_MEAN) / self.F0_STD, reduction='none')
        body_mask = ~sil_mask
        body_mse *= body_mask.type_as(body_mse)  # noqa
        body_mse = body_mse.sum() / body_mask.sum()  # noqa
        body_mae = ((pred_f0_body * self.F0_STD + self.F0_MEAN) - true_f0).abs()
        body_mae *= body_mask.type_as(body_mae)  # noqa
        body_mae = body_mae.sum() / body_mask.sum()  # noqa

        loss = sil_loss + body_mse

        return loss, sil_acc, body_mae

    # def training_step(self, batch, batch_idx):
    #     _, true_audio_len, true_text, text_len, durs, f0 = batch
    #     pred_f0_sil, pred_f0_body = self(text=true_text, text_len=true_text_len)
    #     loss, sil_acc, body_mae = self._metrics(
    #         true_mel_mask=true_mel_mask,
    #     )
    #
    #     return {
    #         'loss': loss,
    #         'progress_bar': {'train_loss': loss, 'train_acc': acc},
    #         'log': {'train_loss': loss, 'train_acc': acc},
    #     }

    def validation_step(self, batch, batch_idx):
        _, _, text, text_len, durs = batch
        durs_pred = self(text=text, text_len=text_len)
        loss = self.loss(durs_true=durs, len_true=text_len, durs_pred=durs_pred)  # noqa
        acc = self._acc(durs_true=durs, len_true=text_len, durs_pred=durs_pred)
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        return {
            'val_loss': avg_loss,
            'log': {'val_loss': avg_loss, 'val_acc': avg_acc},
        }

    @staticmethod
    def _loader(cfg):
        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(  # noqa
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            **cfg.dataloader_params,
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


class TalkNetSpectLoss(Loss):
    """A Loss module that computes loss for TalkNet spectrogram predictor."""

    @property
    def input_types(self):
        return {
            'mel_true': NeuralType(('B', 'D', 'T'), SpectrogramType()),
            'len_true': NeuralType(('B',), LengthsType()),
            'mel_pred': NeuralType(('B', 'D', 'T'), SpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            'loss': NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, *, mel_true, len_true, mel_pred):
        loss = F.mse_loss(mel_pred, mel_true, reduction='none').mean(dim=-2)
        mask = get_mask_from_lengths(len_true)
        loss *= mask.float()
        return loss.sum() / mask.sum()


class TalkNetSpectModel(SpectrogramGenerator):
    """TalkNet's mel spectrogram prediction pipeline."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):  # noqa
        super().__init__(cfg=cfg, trainer=trainer)

        cfg = self._cfg
        self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**cfg.train_ds.dataset.vocab)
        self.preprocessor = instantiate(cfg.preprocessor)
        self.emb = nn.Embedding(len(self.vocab.labels), cfg.d_char)
        self.model = instantiate(cfg.model)
        self.cc = nn.Conv1d(cfg.model.jasper[-1].filters, cfg.n_mels, kernel_size=1)
        self.loss = TalkNetSpectLoss()

    @property
    def input_types(self):
        return {
            'text': NeuralType(('B', 'T'), EmbeddedTextType()),
            'text_len': NeuralType(('B',), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            'mel': NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
        }

    @typecheck()
    def forward(self, *, text, text_len):
        x, x_len = self.emb(text).transpose(1, 2), text_len
        y, _ = self.model(audio_signal=x, length=x_len)
        mel = self.cc(y)
        return mel

    @staticmethod
    def _log(true_mel, true_len, pred_mel):
        loss = F.mse_loss(pred_mel, true_mel, reduction='none').mean(dim=-2)
        mask = get_mask_from_lengths(true_len)
        loss *= mask.float()
        loss = loss.sum() / mask.sum()

        return dict(loss=loss)

    def training_step(self, batch, batch_idx):
        audio, audio_len, text, text_len, _ = batch
        true_mel, _ = self.preprocessor(audio, audio_len)
        pred_mel = self(text=text, text_len=text_len)
        loss = self.loss(mel_true=true_mel, len_true=text_len, mel_pred=pred_mel)  # noqa

        return {
            'loss': loss,
            'progress_bar': {'train_loss': loss},
            'log': {'train_loss': loss},
        }

    def validation_step(self, batch, batch_idx):
        audio, audio_len, text, text_len, _ = batch
        true_mel, _ = self.preprocessor(audio, audio_len)
        pred_mel = self(text=text, text_len=text_len)
        loss = self.loss(mel_true=true_mel, len_true=text_len, mel_pred=pred_mel)  # noqa
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        return {
            'val_loss': avg_loss,
            'log': {'val_loss': avg_loss},
        }

    @staticmethod
    def _loader(cfg):
        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(  # noqa
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            **cfg.dataloader_params,
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
        return torch.tensor(self.vocab.encode(text)).long()

    def generate_spectrogram(self, text: torch.Tensor, **kwargs) -> torch.Tensor:
        text, text_len = text.unsqueeze(0), torch.tensor(len(text)).unsqueeze(0)
        mel = self(text=text, text_len=text_len)[0]
        return mel
