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
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.utils.data as pt_data
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from sacrebleu import corpus_bleu

from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.common.parts import transformer_weights_init
from nemo.collections.common.tokenizers.chinese_tokenizers import ChineseProcessor
from nemo.collections.common.tokenizers.en_ja_tokenizers import EnJaProcessor
from nemo.collections.common.tokenizers.moses_tokenizers import MosesProcessor
from nemo.collections.nlp.data import TarredTranslationDataset, TranslationDataset
from nemo.collections.nlp.models.enc_dec_nlp_model import EncDecNLPModel
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import MTEncDecModelConfig
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.transformer import BeamSearchSequenceGenerator
from nemo.collections.nlp.modules.common.transformer.transformer import TransformerDecoderNM, TransformerEncoderNM
from nemo.core.classes.common import typecheck
from nemo.utils import logging, model_utils

__all__ = ['MTEncDecModel']


class MTEncDecModel(EncDecNLPModel):
    """
    Encoder-decoder machine translation model.
    """

    def __init__(self, cfg: MTEncDecModelConfig, trainer: Trainer = None):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0

        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_gpus

        cfg = model_utils.maybe_update_config_version(cfg)

        self.src_language: str = cfg.get("src_language", None)
        self.tgt_language: str = cfg.get("tgt_language", None)

        # Instantiates tokenizers and register to be saved with NeMo Model archive
        # After this call, ther will be self.encoder_tokenizer and self.decoder_tokenizer
        # Which can convert between tokens and token_ids for SRC and TGT languages correspondingly.
        self.setup_enc_dec_tokenizers(
            encoder_tokenizer_name=cfg.encoder_tokenizer.tokenizer_name,
            encoder_tokenizer_model=cfg.encoder_tokenizer.tokenizer_model,
            encoder_bpe_dropout=cfg.encoder_tokenizer.get('bpe_dropout', 0.0),
            decoder_tokenizer_name=cfg.decoder_tokenizer.tokenizer_name,
            decoder_tokenizer_model=cfg.decoder_tokenizer.tokenizer_model,
            decoder_bpe_dropout=cfg.decoder_tokenizer.get('bpe_dropout', 0.0),
        )

        # After this call, the model will have  self.source_processor and self.target_processor objects
        self.setup_pre_and_post_processing_utils(source_lang=self.src_language, target_lang=self.tgt_language)

        # TODO: Why is this base constructor call so late in the game?
        super().__init__(cfg=cfg, trainer=trainer)

        # TODO: use get_encoder function with support for HF and Megatron
        self.encoder = TransformerEncoderNM(
            vocab_size=self.encoder_vocab_size,
            hidden_size=cfg.encoder.hidden_size,
            num_layers=cfg.encoder.num_layers,
            inner_size=cfg.encoder.inner_size,
            max_sequence_length=cfg.encoder.max_sequence_length
            if hasattr(cfg.encoder, 'max_sequence_length')
            else 512,
            embedding_dropout=cfg.encoder.embedding_dropout if hasattr(cfg.encoder, 'embedding_dropout') else 0.0,
            learn_positional_encodings=cfg.encoder.learn_positional_encodings
            if hasattr(cfg.encoder, 'learn_positional_encodings')
            else False,
            num_attention_heads=cfg.encoder.num_attention_heads,
            ffn_dropout=cfg.encoder.ffn_dropout,
            attn_score_dropout=cfg.encoder.attn_score_dropout,
            attn_layer_dropout=cfg.encoder.attn_layer_dropout,
            hidden_act=cfg.encoder.hidden_act,
            mask_future=cfg.encoder.mask_future,
            pre_ln=cfg.encoder.pre_ln,
        )

        # TODO: user get_decoder function with support for HF and Megatron
        self.decoder = TransformerDecoderNM(
            vocab_size=self.decoder_vocab_size,
            hidden_size=cfg.decoder.hidden_size,
            num_layers=cfg.decoder.num_layers,
            inner_size=cfg.decoder.inner_size,
            max_sequence_length=cfg.decoder.max_sequence_length
            if hasattr(cfg.decoder, 'max_sequence_length')
            else 512,
            embedding_dropout=cfg.decoder.embedding_dropout if hasattr(cfg.decoder, 'embedding_dropout') else 0.0,
            learn_positional_encodings=cfg.decoder.learn_positional_encodings
            if hasattr(cfg.decoder, 'learn_positional_encodings')
            else False,
            num_attention_heads=cfg.decoder.num_attention_heads,
            ffn_dropout=cfg.decoder.ffn_dropout,
            attn_score_dropout=cfg.decoder.attn_score_dropout,
            attn_layer_dropout=cfg.decoder.attn_layer_dropout,
            hidden_act=cfg.decoder.hidden_act,
            pre_ln=cfg.decoder.pre_ln,
        )

        self.log_softmax = TokenClassifier(
            hidden_size=self.decoder.hidden_size,
            num_classes=self.decoder_vocab_size,
            activation=cfg.head.activation,
            log_softmax=cfg.head.log_softmax,
            dropout=cfg.head.dropout,
            use_transformer_init=cfg.head.use_transformer_init,
        )

        self.beam_search = BeamSearchSequenceGenerator(
            embedding=self.decoder.embedding,
            decoder=self.decoder.decoder,
            log_softmax=self.log_softmax,
            max_sequence_length=self.decoder.max_sequence_length,
            beam_size=cfg.beam_size,
            bos=self.decoder_tokenizer.bos_id,
            pad=self.decoder_tokenizer.pad_id,
            eos=self.decoder_tokenizer.eos_id,
            len_pen=cfg.len_pen,
            max_delta_length=cfg.max_generation_delta,
        )

        # tie weights of embedding and softmax matrices
        self.log_softmax.mlp.layer0.weight = self.decoder.embedding.token_embedding.weight

        # TODO: encoder and decoder with different hidden size?
        std_init_range = 1 / self.encoder.hidden_size ** 0.5
        self.apply(lambda module: transformer_weights_init(module, std_init_range))

        self.loss_fn = SmoothedCrossEntropyLoss(
            pad_id=self.decoder_tokenizer.pad_id, label_smoothing=cfg.label_smoothing
        )
        self.eval_loss = GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True)

    def filter_predicted_ids(self, ids):
        ids[ids >= self.decoder_tokenizer.vocab_size] = self.decoder_tokenizer.unk_id
        return ids

    @typecheck()
    def forward(self, src, src_mask, tgt, tgt_mask):
        src_hiddens = self.encoder(src, src_mask)
        tgt_hiddens = self.decoder(tgt, tgt_mask, src_hiddens, src_mask)
        log_probs = self.log_softmax(hidden_states=tgt_hiddens)
        return log_probs

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
        src_ids, src_mask, tgt_ids, tgt_mask, labels = batch
        log_probs = self(src_ids, src_mask, tgt_ids, tgt_mask)
        train_loss = self.loss_fn(log_probs=log_probs, labels=labels)
        tensorboard_logs = {
            'train_loss': train_loss,
            'lr': self._optimizer.param_groups[0]['lr'],
        }
        return {'loss': train_loss, 'log': tensorboard_logs}

    def eval_step(self, batch, batch_idx, mode):
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                batch[i] = batch[i].squeeze(dim=0)
        src_ids, src_mask, tgt_ids, tgt_mask, labels = batch
        log_probs = self(src_ids, src_mask, tgt_ids, tgt_mask)

        # this will run encoder twice -- TODO: potentially fix
        _, translations = self.batch_translate(src=src_ids, src_mask=src_mask)
        eval_loss = self.loss_fn(log_probs=log_probs, labels=labels)
        self.eval_loss(loss=eval_loss, num_measurements=log_probs.shape[0] * log_probs.shape[1])
        np_tgt = tgt_ids.cpu().numpy()
        ground_truths = [self.decoder_tokenizer.ids_to_text(tgt) for tgt in np_tgt]
        ground_truths = [self.target_processor.detokenize(tgt.split(' ')) for tgt in ground_truths]
        num_non_pad_tokens = np.not_equal(np_tgt, self.decoder_tokenizer.pad_id).sum().item()
        return {
            'translations': translations,
            'ground_truths': ground_truths,
            'num_non_pad_tokens': num_non_pad_tokens,
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
        eval_loss = self.eval_loss.compute()
        translations = list(itertools.chain(*[x['translations'] for x in outputs]))
        ground_truths = list(itertools.chain(*[x['ground_truths'] for x in outputs]))

        assert len(translations) == len(ground_truths)
        if self.tgt_language in ['ja']:
            sacre_bleu = corpus_bleu(translations, [ground_truths], tokenize="ja-mecab")
        elif self.tgt_language in ['zh']:
            sacre_bleu = corpus_bleu(translations, [ground_truths], tokenize="zh")
        else:
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

        ans = {f"{mode}_loss": eval_loss, f"{mode}_sacreBLEU": sacre_bleu.score}
        ans['log'] = dict(ans)
        return ans

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        self.log_dict(self.eval_epoch_end(outputs, 'val'), sync_dist=True)

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'test')

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        if cfg.get("load_from_cached_dataset", False):
            logging.info('Loading from cached dataset %s' % (cfg.src_file_name))
            if cfg.src_file_name != cfg.tgt_file_name:
                raise ValueError("src must be equal to target for cached dataset")
            dataset = pickle.load(open(cfg.src_file_name, 'rb'))
            dataset.reverse_lang_direction = cfg.get("reverse_lang_direction", False)
        elif cfg.get("use_tarred_dataset", False):
            if cfg.get('tar_files') is None:
                raise FileNotFoundError("Could not find tarred dataset.")
            logging.info(f'Loading from tarred dataset {cfg.get("tar_files")}')
            if cfg.get("metadata_file", None) is None:
                raise FileNotFoundError("Could not find metadata path in config")
            dataset = TarredTranslationDataset(
                text_tar_filepaths=cfg.tar_files,
                metadata_path=cfg.metadata_file,
                encoder_tokenizer=self.encoder_tokenizer,
                decoder_tokenizer=self.decoder_tokenizer,
                shuffle_n=cfg.get("tar_shuffle_n", 100),
                shard_strategy=cfg.get("shard_strategy", "scatter"),
                global_rank=self.global_rank,
                world_size=self.world_size,
                reverse_lang_direction=cfg.get("reverse_lang_direction", False),
            )
            return torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=1,
                num_workers=cfg.get("num_workers", 2),
                pin_memory=cfg.get("pin_memory", False),
                drop_last=cfg.get("drop_last", False),
            )
        else:
            dataset = TranslationDataset(
                dataset_src=str(Path(cfg.src_file_name).expanduser()),
                dataset_tgt=str(Path(cfg.tgt_file_name).expanduser()),
                tokens_in_batch=cfg.tokens_in_batch,
                clean=cfg.get("clean", False),
                max_seq_length=cfg.get("max_seq_length", 512),
                min_seq_length=cfg.get("min_seq_length", 1),
                max_seq_length_diff=cfg.get("max_seq_length_diff", 512),
                max_seq_length_ratio=cfg.get("max_seq_length_ratio", 512),
                cache_ids=cfg.get("cache_ids", False),
                cache_data_per_node=cfg.get("cache_data_per_node", False),
                use_cache=cfg.get("use_cache", False),
                reverse_lang_direction=cfg.get("reverse_lang_direction", False),
            )
            dataset.batchify(self.encoder_tokenizer, self.decoder_tokenizer)
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

    def setup_pre_and_post_processing_utils(self, source_lang, target_lang):
        """
        Creates source and target processor objects for input and output pre/post-processing.
        """
        self.source_processor, self.target_processor = None, None
        if (source_lang == 'en' and target_lang == 'ja') or (source_lang == 'ja' and target_lang == 'en'):
            self.source_processor = EnJaProcessor(source_lang)
            self.target_processor = EnJaProcessor(target_lang)
        else:
            if source_lang == 'zh':
                self.source_processor = ChineseProcessor()
            if target_lang == 'zh':
                self.target_processor = ChineseProcessor()
            if source_lang is not None and source_lang not in ['ja', 'zh']:
                self.source_processor = MosesProcessor(source_lang)
            if target_lang is not None and target_lang not in ['ja', 'zh']:
                self.target_processor = MosesProcessor(target_lang)

    @torch.no_grad()
    def batch_translate(
        self, src: torch.LongTensor, src_mask: torch.LongTensor,
    ):
        """	
        Translates a minibatch of inputs from source language to target language.	
        Args:	
            src: minibatch of inputs in the src language (batch x seq_len)	
            src_mask: mask tensor indicating elements to be ignored (batch x seq_len)	
        Returns:	
            translations: a list strings containing detokenized translations	
            inputs: a list of string containing detokenized inputs	
        """
        mode = self.training
        try:
            self.eval()

            src_hiddens = self.encoder(input_ids=src, encoder_mask=src_mask)
            beam_results = self.beam_search(encoder_hidden_states=src_hiddens, encoder_input_mask=src_mask)
            beam_results = self.filter_predicted_ids(beam_results)

            translations = [self.decoder_tokenizer.ids_to_text(tr) for tr in beam_results.cpu().numpy()]
            inputs = [self.encoder_tokenizer.ids_to_text(inp) for inp in src.cpu().numpy()]
            if self.target_processor is not None:
                translations = [
                    self.target_processor.detokenize(translation.split(' ')) for translation in translations
                ]

            if self.source_processor is not None:
                inputs = [self.source_processor.detokenize(item.split(' ')) for item in inputs]

        finally:
            self.train(mode=mode)
        return inputs, translations

    # TODO: We should drop source/target_lang arguments in favor of using self.src/tgt_language
    @torch.no_grad()
    def translate(self, text: List[str], source_lang: str = None, target_lang: str = None) -> List[str]:
        """
        Translates list of sentences from source language to target language.
        Should be regular text, this method performs its own tokenization/de-tokenization
        Args:
            text: list of strings to translate
            source_lang: if not None, corresponding MosesTokenizer and MosesPunctNormalizer will be run
            target_lang: if not None, corresponding MosesDecokenizer will be run
        Returns:
            list of translated strings
        """
        # __TODO__: This will reset both source and target processors even if you want to reset just one.
        if source_lang is not None or target_lang is not None:
            self.setup_pre_and_post_processing_utils(source_lang, target_lang)

        mode = self.training
        try:
            self.eval()
            inputs = []
            for txt in text:
                if self.source_processor is not None:
                    txt = self.source_processor.normalize(txt)
                    txt = self.source_processor.tokenize(txt)
                ids = self.encoder_tokenizer.text_to_ids(txt)
                ids = [self.encoder_tokenizer.bos_id] + ids + [self.encoder_tokenizer.eos_id]
                inputs.append(ids)
            max_len = max(len(txt) for txt in inputs)
            src_ids_ = np.ones((len(inputs), max_len)) * self.encoder_tokenizer.pad_id
            for i, txt in enumerate(inputs):
                src_ids_[i][: len(txt)] = txt

            src_mask = torch.FloatTensor((src_ids_ != self.encoder_tokenizer.pad_id)).to(self.device)
            src = torch.LongTensor(src_ids_).to(self.device)
            _, translations = self.batch_translate(src, src_mask)
        finally:
            self.train(mode=mode)
        return translations

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
