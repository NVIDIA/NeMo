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

import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.utils.data as pt_data
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.collections.common.parts import transformer_weights_init
from nemo.collections.nlp.data import TranslationDataset
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.modules.common.transformer import BeamSearchSequenceGenerator, TransformerDecoder, \
    TransformerEmbedding, TransformerEncoder
from nemo.core.classes.common import typecheck
from nemo.core.classes.modelPT import ModelPT

__all__ = ['TransformerMTModel']


class TransformerMTModel(ModelPT):
    """
    Left-to-right Transformer language model.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # shared params for dataset and data loaders
        self.dataset_cfg = cfg.dataset
        self.src_tokenizer = get_tokenizer(
            tokenizer_name=cfg.machine_translation.src_tokenizer,
            tokenizer_model=Path(cfg.machine_translation.src_tokenizer_model).expanduser()
        )
        self.tgt_tokenizer = get_tokenizer(
            tokenizer_name=cfg.machine_translation.tgt_tokenizer,
            tokenizer_model=Path(cfg.machine_translation.tgt_tokenizer_model).expanduser()
        )

        # make vocabulary size divisible by 8 for fast fp16 training
        src_vocab_size = 8 * math.ceil(self.src_tokenizer.vocab_size / 8)
        tgt_vocab_size = 8 * math.ceil(self.tgt_tokenizer.vocab_size / 8)

        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        self.embedding_layer = TransformerEmbedding(
            vocab_size=src_vocab_size,
            hidden_size=cfg.machine_translation.hidden_size,
            max_sequence_length=cfg.machine_translation.max_seq_length,
            embedding_dropout=cfg.machine_translation.get("embedding_dropout", 0.0),
            learn_positional_encodings=False,
        )
        self.encoder = TransformerEncoder(
            hidden_size=cfg.machine_translation.hidden_size,
            inner_size=cfg.machine_translation.inner_size,
            num_layers=cfg.machine_translation.num_layers,
            num_attention_heads=cfg.machine_translation.num_attn_heads,
            ffn_dropout=cfg.machine_translation.ffn_dropout,
            attn_score_dropout=cfg.machine_translation.attn_score_dropout,
            attn_layer_dropout=cfg.machine_translation.attn_layer_dropout,
        )
        self.decoder = TransformerDecoder(
            hidden_size=cfg.machine_translation.hidden_size,
            inner_size=cfg.machine_translation.inner_size,
            num_layers=cfg.machine_translation.num_layers,
            num_attention_heads=cfg.machine_translation.num_attn_heads,
            ffn_dropout=cfg.machine_translation.ffn_dropout,
            attn_score_dropout=cfg.machine_translation.attn_score_dropout,
            attn_layer_dropout=cfg.machine_translation.attn_layer_dropout,
        )
        self.log_softmax = TokenClassifier(
            hidden_size=cfg.machine_translation.hidden_size, num_classes=tgt_vocab_size, log_softmax=True,
        )
        self.beam_search = BeamSearchSequenceGenerator(
            embedding=self.embedding_layer,
            decoder=self.decoder,
            log_softmax=self.log_softmax,
            max_sequence_length=cfg.machine_translation.max_seq_length,
            beam_size=cfg.machine_translation.beam_size,
            bos=self.tgt_tokenizer.bos_id,
            pad=self.tgt_tokenizer.pad_id,
            eos=self.tgt_tokenizer.eos_id,
        )

        std_init_range = 1 / math.sqrt(cfg.machine_translation.hidden_size)
        self.apply(lambda module: transformer_weights_init(module, std_init_range))

        # tie weights of embedding and softmax matrices
        self.log_softmax.mlp.layer0.weight = self.embedding_layer.token_embedding.weight

        self.loss_fn = SmoothedCrossEntropyLoss(
            pad_id=self.tgt_tokenizer.pad_id, label_smoothing=cfg.machine_translation.label_smoothing)

        self.last_eval_loss = None
        self.last_eval_beam_results = None

        # Optimizer setup needs to happen after all model weights are ready
        self.setup_optimization(cfg.optim)

    @typecheck()
    def forward(self, src, src_mask, tgt, tgt_mask):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        if src.ndim == 3:
            # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
            # is excess.
            src = src.squeeze(dim=0)
            src_mask = src_mask.squeeze(dim=0)
            tgt = tgt.squeeze(dim=0)
            tgt_mask = tgt_mask.squeeze(dim=0)
        src_embeddings = self.embedding_layer(input_ids=src)
        src_hiddens = self.encoder(src_embeddings, src_mask)
        tgt_embeddings = self.embedding_layer(input_ids=tgt)
        tgt_hiddens = self.decoder(tgt_embeddings, tgt_mask, src_hiddens, src_mask)
        log_probs = self.log_softmax(hidden_states=tgt_hiddens)
        beam_results = None
        if not self.training:
            beam_results = self.beam_search(
                encoder_hidden_states=src_hiddens,
                encoder_input_mask=src_mask)

        return log_probs, beam_results

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        src_ids, src_mask, tgt_ids, tgt_mask, labels, _ = batch
        log_probs, _ = self(src_ids, src_mask, tgt_ids, tgt_mask)

        if labels.ndim == 3:
            # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
            # is excess.
            labels = labels.squeeze(dim=0)
        train_loss = self.loss_fn(log_probs=log_probs, labels=labels)

        tensorboard_logs = {'train_loss': train_loss, 'lr': self._optimizer.param_groups[0]['lr']}
        return {'loss': train_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        src_ids, src_mask, tgt_ids, tgt_mask, labels, sent_ids = batch
        try:
            log_probs, beam_results = self(src_ids, src_mask, tgt_ids, tgt_mask)
        except IndexError as e:
            print("(TransformerMTModel.validation_step)src_ids.shape:", src_ids.shape)
            raise e
        if labels.ndim == 3:
            # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
            # is excess.
            labels = labels.squeeze(dim=0)
            sent_ids = sent_ids.squeeze(dim=0)
        val_loss = self.loss_fn(log_probs=log_probs, labels=labels)
        self.last_eval_beam_results = beam_results
        self.last_eval_loss = val_loss

        tensorboard_logs = {'val_loss': val_loss}

        return {'val_loss': val_loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_ppl': torch.exp(avg_loss)}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        dataset = TranslationDataset(
            tokenizer_src=self.src_tokenizer,
            tokenizer_tgt=self.tgt_tokenizer,
            dataset_src=str(Path(cfg.src_file_name).expanduser()),
            dataset_tgt=str(Path(cfg.tgt_file_name).expanduser()),
            tokens_in_batch=cfg.tokens_in_batch,
        )
        if cfg.shuffle:
            sampler = pt_data.RandomSampler(dataset)
        else:
            sampler = pt_data.SequentialSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=self.dataset_cfg.get("num_workers", 2),
            pin_memory=self.dataset_cfg.get("pin_memory", False),
            drop_last=self.dataset_cfg.get("drop_last", False),
        )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
