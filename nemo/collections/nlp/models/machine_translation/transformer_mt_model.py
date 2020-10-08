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
            d_inner=cfg.machine_translation.d_inner,
            num_layers=cfg.machine_translation.num_layers,
            embedding_dropout=cfg.machine_translation.embedding_dropout,
            num_attn_heads=cfg.machine_translation.num_attn_heads,
            ffn_dropout=cfg.machine_translation.ffn_dropout,
            vocab_size=self.src_tokenizer.vocab_size,
            attn_score_dropout=cfg.machine_translation.attn_score_dropout,
            attn_layer_dropout=cfg.machine_translation.attn_layer_dropout,
            max_seq_length=cfg.machine_translation.max_seq_length,
        )
        self.decoder = TransformerDecoder(
            hidden_size=cfg.machine_translation.hidden_size,
            d_inner=cfg.machine_translation.d_inner,
            num_layers=cfg.machine_translation.num_layers,
            embedding_dropout=cfg.machine_translation.embedding_dropout,
            num_attn_heads=cfg.machine_translation.num_attn_heads,
            ffn_dropout=cfg.machine_translation.ffn_dropout,
            vocab_size=self.tgt_tokenizer.vocab_size,
            attn_score_dropout=cfg.machine_translation.attn_score_dropout,
            attn_layer_dropout=cfg.machine_translation.attn_layer_dropout,
            max_seq_length=cfg.machine_translation.max_seq_length,
        )
        self.log_softmax = TokenClassifier(
            hidden_size=cfg.machine_translation.hidden_size, num_classes=tgt_vocab_size, log_softmax=True,
        )
        self.beam_search = BeamSearchSequenceGenerator(
            embedding=self.embedding_layer,
            decoder=self.decoder,
            log_softmax=self.log_softmax,
            max_seq_length=cfg.machine_translation.max_seq_length,
            beam_size=cfg.machine_translation.beam_size,
            bos_token=self.tgt_tokenizer.bos_id,
            pad_token=self.tgt_tokenizer.pad_id,
            eos_token=self.tgt_tokenizer.eos_id,
        )

        std_init_range = 1 / math.sqrt(cfg.machine_translation.hidden_size)
        self.apply(lambda module: transformer_weights_init(module, std_init_range))

        # tie weights of embedding and softmax matrices
        self.log_softmax.mlp.layer0.weight = self.embedding_layer.token_embedding.weight

        self.loss_fn = SmoothedCrossEntropyLoss(
            pad_id=self.tokenizer.pad_id, label_smoothing=cfg.machine_translation.label_smoothing)

        # Optimizer setup needs to happen after all model weights are ready
        self.setup_optimization(cfg.optim)

    @typecheck()
    def forward(self, input_ids, attention_mask):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        token_embeddings = self.embedding_layer(input_ids)
        hidden_states = self.encoder(token_embeddings, attention_mask)
        decoder_hidden_states = self.decoder(hidden_states)
        log_probs = self.log_softmax(hidden_states=decoder_hidden_states)

        return log_probs

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        input_ids, input_mask, labels = batch
        log_probs = self(input_ids=input_ids, attention_mask=input_mask)

        train_loss = self.loss_fn(logits=log_probs, labels=labels)

        tensorboard_logs = {'train_loss': train_loss, 'lr': self._optimizer.param_groups[0]['lr']}
        return {'loss': train_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_mask, labels = batch
        log_probs = self(input_ids=input_ids, attention_mask=input_mask)

        val_loss = self.loss_fn(logits=log_probs, labels=labels)

        tensorboard_logs = {
            'val_loss': val_loss,
        }

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
