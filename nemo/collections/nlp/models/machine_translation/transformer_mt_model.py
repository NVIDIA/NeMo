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
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.utils.data as pt_data
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.collections.nlp.metrics.sacrebleu import corpus_bleu
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
        if "tokenizer" in cfg.machine_translation:
            if "src_tokenizer" in cfg.machine_translation or "tgt_tokenizer" in cfg.machine_translation:
                raise ValueError(
                    "If 'tokenizer' is in 'machine_translation' section of config then this section should "
                    "not contain 'src_tokenizer' and 'tgt_tokenizer' fields.")
            self.src_tokenizer = get_tokenizer(**cfg.machine_translation.tokenizer)
            self.tgt_tokenizer = self.src_tokenizer
            super().__init__(cfg=cfg, trainer=trainer)
            # make vocabulary size divisible by 8 for fast fp16 training
            src_vocab_size = 8 * math.ceil(self.src_tokenizer.vocab_size / 8)
            tgt_vocab_size = src_vocab_size
            self.src_embedding_layer = TransformerEmbedding(
                vocab_size=src_vocab_size,
                hidden_size=cfg.machine_translation.hidden_size,
                max_sequence_length=cfg.machine_translation.max_seq_length,
                embedding_dropout=cfg.machine_translation.get("embedding_dropout", 0.0),
                learn_positional_encodings=False,
            )
            self.tgt_embedding_layer = self.src_embedding_layer
        else:
            self.src_tokenizer = get_tokenizer(**cfg.machine_translation.src_tokenizer)
            self.tgt_tokenizer = get_tokenizer(**cfg.machine_translation.tgt_tokenizer)
            super().__init__(cfg=cfg, trainer=trainer)
            # make vocabulary size divisible by 8 for fast fp16 training
            src_vocab_size = 8 * math.ceil(self.src_tokenizer.vocab_size / 8)
            tgt_vocab_size = 8 * math.ceil(self.tgt_tokenizer.vocab_size / 8)
            self.src_embedding_layer = TransformerEmbedding(
                vocab_size=src_vocab_size,
                hidden_size=cfg.machine_translation.hidden_size,
                max_sequence_length=cfg.machine_translation.max_seq_length,
                embedding_dropout=cfg.machine_translation.get("embedding_dropout", 0.0),
                learn_positional_encodings=False,
            )
            self.tgt_embedding_layer = TransformerEmbedding(
                vocab_size=tgt_vocab_size,
                hidden_size=cfg.machine_translation.hidden_size,
                max_sequence_length=cfg.machine_translation.max_seq_length,
                embedding_dropout=cfg.machine_translation.get("embedding_dropout", 0.0),
                learn_positional_encodings=False,
            )
            
        # init superclass
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
            embedding=self.tgt_embedding_layer,
            decoder=self.decoder,
            log_softmax=self.log_softmax,
            max_sequence_length=cfg.machine_translation.max_seq_length,
            beam_size=cfg.machine_translation.beam_size,
            bos=self.tgt_tokenizer.bos_id,
            pad=self.tgt_tokenizer.pad_id,
            eos=self.tgt_tokenizer.eos_id,
            len_pen=cfg.machine_translation.len_pen,
        )

        std_init_range = 1 / math.sqrt(cfg.machine_translation.hidden_size)
        self.apply(lambda module: transformer_weights_init(module, std_init_range))

        # tie weights of embedding and softmax matrices
        self.log_softmax.mlp.layer0.weight = self.tgt_embedding_layer.token_embedding.weight

        self.loss_fn = SmoothedCrossEntropyLoss(
            pad_id=self.tgt_tokenizer.pad_id, label_smoothing=cfg.machine_translation.label_smoothing)

        # Optimizer setup needs to happen after all model weights are ready
        self.setup_optimization(cfg.optim)

    @typecheck()
    def forward(self, src, src_mask, tgt, tgt_mask):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        src_embeddings = self.src_embedding_layer(input_ids=src)
        src_hiddens = self.encoder(src_embeddings, src_mask)
        tgt_embeddings = self.tgt_embedding_layer(input_ids=tgt)
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
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                batch[i] = batch[i].squeeze(dim=0)
        src_ids, src_mask, tgt_ids, tgt_mask, labels, _ = batch
        log_probs, _ = self(src_ids, src_mask, tgt_ids, tgt_mask)
        train_loss = self.loss_fn(log_probs=log_probs, labels=labels)

        tensorboard_logs = {'train_loss': train_loss, 'lr': self._optimizer.param_groups[0]['lr']}
        return {'loss': train_loss, 'log': tensorboard_logs}

    def eval_step(self, batch, batch_idx, mode):
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                batch[i] = batch[i].squeeze(dim=0)
        src_ids, src_mask, tgt_ids, tgt_mask, labels, sent_ids = batch
        log_probs, beam_results = self(src_ids, src_mask, tgt_ids, tgt_mask)
        eval_loss = self.loss_fn(log_probs=log_probs, labels=labels).cpu().numpy()
        translations = [self.tgt_tokenizer.ids_to_text(tr) for tr in beam_results.cpu().numpy()]
        np_tgt = tgt_ids.cpu().numpy()
        ground_truths = [self.tgt_tokenizer.ids_to_text(tgt) for tgt in np_tgt]
        num_non_pad_tokens = np.not_equal(np_tgt, self.tgt_tokenizer.pad_id).sum().item()
        tensorboard_logs = {f'{mode}_loss': eval_loss}
        return {
            f'{mode}_loss': eval_loss,
            'translations': translations,
            'ground_truths': ground_truths,
            'num_non_pad_tokens': num_non_pad_tokens,
            'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        return self.eval_step(batch, batch_idx, 'val')

    def eval_epoch_end(self, outputs, mode):
        counts = np.array([x['num_non_pad_tokens'] for x in outputs])
        eval_loss = np.sum(np.array([x[f'{mode}_loss'] for x in outputs]) * counts) / counts.sum()
        translations = list(itertools.chain(*[x['translations'] for x in outputs]))
        ground_truths = list(itertools.chain(*[x['ground_truths'] for x in outputs]))
        token_bleu = corpus_bleu(translations, [ground_truths], tokenize="fairseq")
        sacre_bleu = corpus_bleu(translations, [ground_truths], tokenize="13a")
        ans = {
            f"{mode}_loss": eval_loss, 
            f"{mode}_tokenBLEU": token_bleu.score, 
            f"{mode}_sacreBLEU": sacre_bleu.score
        }
        ans['log'] = dict(ans)
        return ans

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        return self.eval_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'test')

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
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
