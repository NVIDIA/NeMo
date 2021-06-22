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

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from nemo.collections.asr.data.audio_to_text import AudioToCharWithDursF0Dataset
from nemo.collections.tts.helpers.helpers import binarize_attention, get_mask_from_lengths, plot_alignment_to_numpy
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.core.classes import ModelPT
from nemo.core.classes.common import typecheck
from nemo.utils import logging

HAVE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    HAVE_WANDB = False


class AlignerModel(ModelPT):
    """Speech-to-text alignment pipeline."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)
        typecheck.set_typecheck_enabled(enabled=False)

        cfg = self._cfg
        self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**cfg.train_ds.dataset.vocab)
        self.embed = nn.Embedding(len(self.vocab.labels), cfg.d_char)
        self.preprocessor = instantiate(cfg.preprocessor)
        self.alignment_encoder = instantiate(cfg.alignment_encoder)

        self.forward_sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()

        self.bin_start_ratio = cfg.bin_start_ratio
        self.add_bin_loss = False

    def forward(self, *, spec, spec_len, text, text_len, attn_prior=None):
        with torch.cuda.amp.autocast(enabled=False):
            attn_soft, attn_logprob = self.alignment_encoder(
                queries=spec,
                keys=self.embed(text).transpose(1, 2),
                mask=get_mask_from_lengths(text_len).unsqueeze(-1) == 0,
                attn_prior=attn_prior,
            )

        return attn_soft, attn_logprob

    def _metrics(self, attn_soft, attn_logprob, spec_len, text_len):
        loss, bin_loss, attn_hard = 0.0, 0.0, None

        forward_sum_loss = self.forward_sum_loss(attn_logprob, text_len, spec_len)
        loss += forward_sum_loss

        if self.add_bin_loss:
            attn_hard = binarize_attention(attn_soft, text_len, spec_len)
            bin_loss = self.bin_loss(attn_hard, attn_soft)
            loss += bin_loss

        return loss, forward_sum_loss, bin_loss, attn_hard

    def on_train_epoch_start(self):
        # Add bin loss when current_epoch >= bin_start_ratio * max_epochs
        if not self.add_bin_loss and self.current_epoch >= np.ceil(self.bin_start_ratio * self._trainer.max_epochs):
            logging.info(f"Using hard attentions after epoch: {self.current_epoch}")
            self.add_bin_loss = True

    def training_step(self, batch, batch_idx):
        audio, audio_len, text, text_len, attn_prior = batch
        spec, spec_len = self.preprocessor(audio, audio_len)
        attn_soft, attn_logprob = self(
            spec=spec, spec_len=spec_len, text=text, text_len=text_len, attn_prior=attn_prior
        )

        loss, forward_sum_loss, bin_loss, _ = self._metrics(attn_soft, attn_logprob, spec_len, text_len)

        train_log = {'train_forward_sum_loss': forward_sum_loss, 'train_bin_loss': bin_loss}
        return {'loss': loss, 'progress_bar': train_log, 'log': train_log}

    def validation_step(self, batch, batch_idx):
        audio, audio_len, text, text_len, attn_prior = batch
        spec, spec_len = self.preprocessor(audio, audio_len)
        attn_soft, attn_logprob = self(
            spec=spec, spec_len=spec_len, text=text, text_len=text_len, attn_prior=attn_prior
        )

        loss, forward_sum_loss, bin_loss, attn_hard = self._metrics(attn_soft, attn_logprob, spec_len, text_len)

        # plot once per epoch
        if batch_idx == 0 and isinstance(self.logger, WandbLogger) and HAVE_WANDB:
            if attn_hard is None:
                attn_hard = binarize_attention(attn_soft, text_len, spec_len)

            attn_matrices = []
            for i in range(min(5, audio.shape[0])):
                attn_matrices.append(
                    wandb.Image(
                        plot_alignment_to_numpy(
                            np.fliplr(np.rot90(attn_soft[i, 0, : spec_len[i], : text_len[i]].data.cpu().numpy()))
                        ),
                        caption=f"attn soft",
                    ),
                )

                attn_matrices.append(
                    wandb.Image(
                        plot_alignment_to_numpy(
                            np.fliplr(np.rot90(attn_hard[i, 0, : spec_len[i], : text_len[i]].data.cpu().numpy()))
                        ),
                        caption=f"attn hard",
                    )
                )

            self.logger.experiment.log({"attn_matrices": attn_matrices})

        val_log = {'val_loss': loss, 'val_forward_sum_loss': forward_sum_loss, 'val_bin_loss': bin_loss}
        self.log_dict(val_log, prog_bar=False, on_epoch=True, logger=True, sync_dist=True)

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
