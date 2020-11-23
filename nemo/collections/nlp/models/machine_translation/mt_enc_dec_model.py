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

from dataclasses import dataclass
import itertools
import math
from nemo.collections.nlp.models.enc_dec_nlp_model import (
    EmbeddingConfig,
    EncDecNLPModelConfig,
    EncDecNLPModel,
    TransformerEmbeddingConfig,
)
import random
from pathlib import Path
from typing import Dict, List, Optional, Union
from hydra.utils import instantiate

import numpy as np
import torch
import torch.utils.data as pt_data
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from sacrebleu import corpus_bleu

from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.collections.common.metrics import Perplexity
from nemo.collections.common.parts import transformer_weights_init
from nemo.collections.nlp.data import TranslationDataset
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.modules.common.transformer import (
    BeamSearchSequenceGenerator,
    TransformerDecoder,
    TransformerEmbedding,
    TransformerEncoder,
)
from nemo.core.classes.common import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.utils import logging

__all__ = ['MTEncDecModel']


@dataclass
class MTEncDecModelConfig(EncDecNLPModelConfig):
    num_val_examples: int = 3
    num_test_examples: int = 3


class MTEncDecModel(EncDecNLPModel):
    """
    Left-to-right Transformer language model.
    """

    def __init__(self, cfg: MTEncDecModelConfig, trainer: Trainer = None):

        self.setup_enc_dec_tokenizers(cfg)

        super().__init__(cfg=cfg, trainer=trainer)

        # make vocabulary size divisible by 8 for fast fp16 training
        if cfg.vocab_divisibile_by_eight:
            self.enc_vocab_size = 8 * math.ceil(self.enc_tokenizer.vocab_size / 8)
            self.dec_vocab_size = 8 * math.ceil(self.dec_tokenizer.vocab_size / 8)
            cfg.enc_embedding.vocab_size = self.enc_vocab_size
            cfg.dec_embedding.vocab_size = self.dec_vocab_size

        self.enc_embedding = instantiate(cfg.enc_embedding)
        self.dec_embedding = instantiate(cfg.dec_embedding)

        # self.enc_embedding = TransformerEmbedding(
        #     vocab_size=cfg.enc_embedding.vocab_size,
        #     hidden_size=cfg.enc_embedding.hidden_size,
        #     max_sequence_length=cfg.enc_embedding.max_sequence_length,
        #     embedding_dropout=cfg.enc_embedding.embedding_dropout,
        #     learn_positional_encodings=cfg.enc_embedding.learn_positional_encodings,
        # )

        # self.enc_embedding = TransformerEmbedding(
        #     vocab_size=self.enc_vocab_size,
        #     hidden_size=cfg.machine_translation.hidden_size,
        #     max_sequence_length=cfg.machine_translation.max_seq_length,
        #     embedding_dropout=cfg.machine_translation.get("embedding_dropout", 0.0),
        #     learn_positional_encodings=False,
        # )
        # self.dec_embedding = TransformerEmbedding(
        #     vocab_size=self.dec_vocab_size,
        #     hidden_size=cfg.machine_translation.hidden_size,
        #     max_sequence_length=cfg.machine_translation.max_seq_length,
        #     embedding_dropout=cfg.machine_translation.get("embedding_dropout", 0.0),
        #     learn_positional_encodings=False,
        # )

        # TODO: Optionally tie Embedding weights

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
            embedding=self.dec_embedding,
            decoder=self.decoder,
            log_softmax=self.log_softmax,
            max_sequence_length=cfg.machine_translation.max_seq_length,
            beam_size=cfg.machine_translation.beam_size,
            bos=self.dec_tokenizer.bos_id,
            pad=self.dec_tokenizer.pad_id,
            eos=self.dec_tokenizer.eos_id,
            len_pen=cfg.machine_translation.len_pen,
            max_delta_length=cfg.machine_translation.get("max_generation_delta", 50),
        )

        std_init_range = 1 / cfg.machine_translation.hidden_size ** 0.5
        self.apply(lambda module: transformer_weights_init(module, std_init_range))

        # tie weights of embedding and softmax matrices
        self.log_softmax.mlp.layer0.weight = self.dec_embedding.token_embedding.weight
        self.emb_scale = cfg.machine_translation.hidden_size ** 0.5
        self.loss_fn = SmoothedCrossEntropyLoss(
            pad_id=self.dec_tokenizer.pad_id, label_smoothing=cfg.machine_translation.label_smoothing
        )

        # Optimizer setup needs to happen after all model weights are ready
        self.setup_optimization(cfg.optim)

        self.training_perplexity = Perplexity(dist_sync_on_step=True)
        self.eval_perplexity = Perplexity(compute_on_step=False)

        # These attributes are added to bypass Illegal memory access error in PT1.6
        # https://github.com/pytorch/pytorch/issues/21819

    def filter_predicted_ids(self, ids):
        ids[ids >= self.dec_tokenizer.vocab_size] = self.dec_tokenizer.unk_id
        return ids

    @typecheck()
    def forward(self, src, src_mask, tgt, tgt_mask):
        """
        torch.nn.Module.forward method.
        Args:
            src: source ids
            src_mask: src mask (mask padding)
            tgt: target ids
            tgt_mask: target mask

        Returns:

        """
        src_embeddings = self.enc_embedding(input_ids=src)
        # src_embeddings *= src_embeddings.new_tensor(self.emb_scale)
        src_hiddens = self.encoder(src_embeddings, src_mask)
        tgt_embeddings = self.dec_embedding(input_ids=tgt)
        # tgt_embeddings *= tgt_embeddings.new_tensor(self.emb_scale)
        tgt_hiddens = self.decoder(tgt_embeddings, tgt_mask, src_hiddens, src_mask)
        log_probs = self.log_softmax(hidden_states=tgt_hiddens)
        beam_results = None
        if not self.training:
            beam_results = self.beam_search(encoder_hidden_states=src_hiddens, encoder_input_mask=src_mask)
            beam_results = self.filter_predicted_ids(beam_results)
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
        training_perplexity = self.training_perplexity(logits=log_probs)
        tensorboard_logs = {
            'train_loss': train_loss,
            'lr': self._optimizer.param_groups[0]['lr'],
            "train_ppl": training_perplexity,
        }
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
        self.eval_perplexity(logits=log_probs)
        translations = [self.dec_tokenizer.ids_to_text(tr) for tr in beam_results.cpu().numpy()]
        np_tgt = tgt_ids.cpu().numpy()
        ground_truths = [self.dec_tokenizer.ids_to_text(tgt) for tgt in np_tgt]
        num_non_pad_tokens = np.not_equal(np_tgt, self.dec_tokenizer.pad_id).sum().item()
        tensorboard_logs = {f'{mode}_loss': eval_loss}
        return {
            f'{mode}_loss': eval_loss,
            'translations': translations,
            'ground_truths': ground_truths,
            'num_non_pad_tokens': num_non_pad_tokens,
            'log': tensorboard_logs,
        }

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    @rank_zero_only
    def log_param_stats(self):
        for name, p in self.named_parameters():
            if p.requires_grad:
                self.trainer.logger.experiment.add_histogram(name + '_hist', p, global_step=self.global_step)
                self.trainer.logger.experiment.add_scalars(
                    name,
                    {'mean': p.mean(), 'stddev': p.std(), 'max': p.max(), 'min': p.min()},
                    global_step=self.global_step,
                )

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        return self.eval_step(batch, batch_idx, 'val')

    def eval_epoch_end(self, outputs, mode):
        counts = np.array([x['num_non_pad_tokens'] for x in outputs])
        eval_loss = np.sum(np.array([x[f'{mode}_loss'] for x in outputs]) * counts) / counts.sum()
        eval_perplexity = self.eval_perplexity.compute()
        translations = list(itertools.chain(*[x['translations'] for x in outputs]))
        ground_truths = list(itertools.chain(*[x['ground_truths'] for x in outputs]))
        assert len(translations) == len(ground_truths)
        sacre_bleu = corpus_bleu(translations, [ground_truths], tokenize="13a")
        dataset_name = "Validation" if mode == 'val' else "Test"
        logging.info(f"\n\n\n\n{dataset_name} set size: {len(translations)}")
        logging.info(f"{dataset_name} Sacre BLEU = {sacre_bleu.score}")
        logging.info(f"{dataset_name} TRANSLATION EXAMPLES:".upper())
        for i in range(0, 3):
            ind = random.randint(0, len(translations) - 1)
            logging.info("    " + '\u0332'.join(f"EXAMPLE {i}:"))
            logging.info(f"    Prediction:   {translations[ind]}")
            logging.info(f"    Ground Truth: {ground_truths[ind]}")

        ans = {f"{mode}_loss": eval_loss, f"{mode}_sacreBLEU": sacre_bleu.score, f"{mode}_ppl": eval_perplexity}
        ans['log'] = dict(ans)
        return ans

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        self.log_dict(self.eval_epoch_end(outputs, 'val'))
        # return self.eval_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'test')

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)
        self.num_examples['val'] = val_data_config.get('num_examples', self.num_examples['val'])

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)
        self.num_examples['test'] = test_data_config.get('num_examples', self.num_examples['test'])

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        dataset = TranslationDataset(
            tokenizer_src=self.enc_tokenizer,
            tokenizer_tgt=self.dec_tokenizer,
            dataset_src=str(Path(cfg.src_file_name).expanduser()),
            dataset_tgt=str(Path(cfg.tgt_file_name).expanduser()),
            tokens_in_batch=cfg.tokens_in_batch,
            clean=cfg.get("clean", False),
            max_seq_length=cfg.get("max_seq_length", 512),
            min_seq_length=cfg.get("min_seq_length", 1),
            max_seq_length_diff=cfg.get("max_seq_length_diff", 512),
            max_seq_length_ratio=cfg.get("max_seq_length_ratio", 512),
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

    @torch.no_grad()
    def translate(self, text: List[str]) -> List[str]:
        """
        Translates list of sentences from source language to target language.
        Should be regular text, this method performs its own tokenization/de-tokenization
        Args:
            text: list of strings to translate

        Returns:
            list of translated strings
        """
        mode = self.training
        try:
            self.eval()
            res = []
            for txt in text:
                ids = self.enc_tokenizer.text_to_ids(txt)
                ids = [self.enc_tokenizer.bos_id] + ids + [self.enc_tokenizer.eos_id]
                src = torch.Tensor(ids).long().to(self._device).unsqueeze(0)
                src_mask = torch.ones_like(src)
                src_embeddings = self.enc_embedding(input_ids=src)
                # src_embeddings *= src_embeddings.new_tensor(self.emb_scale)
                src_hiddens = self.encoder(src_embeddings, src_mask)
                beam_results = self.beam_search(encoder_hidden_states=src_hiddens, encoder_input_mask=src_mask)
                beam_results = self.filter_predicted_ids(beam_results)
                translation_ids = beam_results.cpu()[0].numpy()
                res.append(self.dec_tokenizer.ids_to_text(translation_ids))
        finally:
            self.train(mode=mode)
        return res

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
