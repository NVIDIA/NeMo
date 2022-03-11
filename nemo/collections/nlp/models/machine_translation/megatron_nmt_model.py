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

import torch
from operator import itemgetter
from typing import Optional
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel
from omegaconf.dictconfig import DictConfig, ListConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import build_train_valid_test_datasets
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.models.language_modeling.megatron_lm_encoder_decoder_model import (
    MegatronLMEncoderDecoderModel,
)

from nemo.utils import logging

try:
    from apex.transformer import tensor_parallel
    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ["MegatronNMTModel"]


class MegatronNMTModel(MegatronLMEncoderDecoderModel):
    """
    Megatron NMT training
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.src_language = cfg.get("src_language", None)
        self.tgt_language = cfg.get("tgt_language", None)

        self.multilingual = cfg.get("multilingual", False)
        self.multilingual_ids = []
        self.special_tokens = {}

        self.encoder_tokenizer_library = cfg.encoder_tokenizer.get('library', 'yttm')
        self.decoder_tokenizer_library = cfg.decoder_tokenizer.get('library', 'yttm')

        self.validate_input_ids = cfg.get("validate_input_ids", True)
        if self.multilingual:
            if isinstance(self.src_language, ListConfig) and isinstance(self.tgt_language, ListConfig):
                raise ValueError(
                    "cfg.src_language and cfg.tgt_language cannot both be lists. We only support many-to-one or one-to-many multilingual models."
                )
            elif isinstance(self.src_language, ListConfig):
                pass
            elif isinstance(self.tgt_language, ListConfig):
                for lng in self.tgt_language:
                    self.special_tokens["<" + lng + ">"] = "<" + lng + ">"
            else:
                raise ValueError(
                    "Expect either cfg.src_language or cfg.tgt_language to be a list when multilingual=True."
                )

    def _build_tokenizer(self):
        # Instantiates tokenizers and register to be saved with NeMo Model archive
        # After this call, ther will be self.encoder_tokenizer and self.decoder_tokenizer
        # Which can convert between tokens and token_ids for SRC and TGT languages correspondingly.
        encoder_tokenizer_model = self.register_artifact("encoder_tokenizer.tokenizer_model", self._cfg.encoder_tokenizer.get('tokenizer_model'))
        decoder_tokenizer_model = self.register_artifact("decoder_tokenizer.tokenizer_model", self._cfg.decoder_tokenizer.get('tokenizer_model'))

        self.encoder_tokenizer, self.decoder_tokenizer = MTEncDecModel.setup_enc_dec_tokenizers(
            encoder_tokenizer_library=self.encoder_tokenizer_library,
            encoder_tokenizer_model=encoder_tokenizer_model,
            encoder_bpe_dropout=self._cfg.encoder_tokenizer.get('bpe_dropout', 0.0)
            if self._cfg.encoder_tokenizer.get('bpe_dropout', 0.0) is not None
            else 0.0,
            encoder_model_name=self._cfg.encoder.get('model_name') if hasattr(self._cfg.encoder, 'model_name') else None,
            encoder_r2l=self._cfg.encoder_tokenizer.get('r2l', False),
            decoder_tokenizer_library=self.decoder_tokenizer_library,
            encoder_tokenizer_vocab_file=self._cfg.encoder_tokenizer.get('vocab_file', None),
            decoder_tokenizer_model=decoder_tokenizer_model,
            decoder_bpe_dropout=self._cfg.decoder_tokenizer.get('bpe_dropout', 0.0)
            if self._cfg.decoder_tokenizer.get('bpe_dropout', 0.0) is not None
            else 0.0,
            decoder_model_name=self._cfg.decoder.get('model_name') if hasattr(self._cfg.decoder, 'model_name') else None,
            decoder_r2l=self._cfg.decoder_tokenizer.get('r2l', False),
            special_tokens=self.special_tokens,
        )

        # Set up pre and post processors as well.
        if self.multilingual:
            self.source_processor_list, self.target_processor_list, self.multilingual_ids = MTEncDecModel.setup_multilingual_ids_and_processors(
                self.src_language,
                self.tgt_language,
                self.encoder_tokenizer,
            )
        else:
            # After this call, the model will have  self.source_processor and self.target_processor objects
            MTEncDecModel.setup_pre_and_post_processing_utils(
                self.src_language, self.tgt_language,
                self.encoder_tokenizer_library, self.decoder_tokenizer_library,
            )
            self.multilingual_ids = [None]

    def validation_step(self, batch, batch_idx):
        """
        Validation step
        """
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask = self.process_batch(batch)
        tokens_loss = itemgetter("tokens_loss")(
            self(tokens_enc, tokens_dec, enc_mask, dec_mask, tokentype_ids=None, lm_labels=labels,)
        )
        predicted_tokens_ids, _ = self.decode(tokens_enc, enc_mask, self._cfg.data.seq_length_dec)
        preds = predicted_tokens_ids.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        outputs = []
        for _, (pred, label) in enumerate(zip(preds, labels)):
            if self.model.tokenizer.eos_id in pred:
                idx = pred.index(self.model.tokenizer.eos_id)
                pred = pred[:idx]

            # Legacy sentencepiece detokenization still preserves special tokens which messes up exact string match.
            if hasattr(self.model.tokenizer, 'special_token_to_id'):
                pred = [id for id in pred if id not in self.model.tokenizer.special_token_to_id.values()]
                label = [id for id in label if id not in self.model.tokenizer.special_token_to_id.values()]
            pred = self.model.tokenizer.ids_to_text(pred)
            outputs.append(pred)

        loss = self.loss_func(loss_mask, tokens_loss)
        reduced_loss = average_losses_across_data_parallel_group([loss])
        return reduced_loss

    def _setup_eval_dataloader_from_config(
        self, cfg: DictConfig,
    ):
        return MTEncDecModel._setup_eval_dataloader_from_config(
            cfg=cfg,
            encoder_tokenizer=self.encoder_tokenizer,
            decoder_tokenizer=self.decoder_tokenizer,
            is_multilingual=self.multilingual,
            multilingual_ids=self.multilingual_ids,
        )

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_eval_dataloader_from_config(
            cfg=val_data_config,
            multilingual=self.multilingual,
            multilingual_ids=self.multilingual_ids,
            encoder_tokenizer=self.encoder_tokenizer,
            decoder_tokenizer=self.decoder_tokenizer
        )

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_eval_dataloader_from_config(
            cfg=test_data_config,
            multilingual=self.multilingual,
            multilingual_ids=self.multilingual_ids,
            encoder_tokenizer=self.encoder_tokenizer,
            decoder_tokenizer=self.decoder_tokenizer
        )

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = MTEncDecModel._setup_dataloader_from_config(
            cfg=train_data_config,
            encoder_tokenizer=self.encoder_tokenizer,
            decoder_tokenizer=self.decoder_tokenizer,
            global_rank=self.global_rank,
            world_size=self.world_size,
            multilingual=self.multilingual,
            multilingual_ids=self.multilingual_ids,
        )

    def process_batch(self, batch):
        """Override parent process_batch since TranslationDataset does not return dictionaries."""
        src_ids, src_mask, tgt_ids, tgt_mask, labels = batch
        batch = {
            'text_enc': src_ids,
            'text_dec': tgt_ids,
            'labels': labels,
            'enc_mask': src_mask,
            'dec_mask': tgt_mask,
            'loss_mask': tgt_mask,
        }
        keys = ['text_enc', 'text_dec', 'labels', 'loss_mask', 'enc_mask', 'dec_mask']
        datatype = torch.int64
        data = batch
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_enc = data_b['text_enc'].long()
        tokens_dec = data_b['text_dec'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()

        enc_mask = data_b['enc_mask']
        dec_mask = data_b['dec_mask']

        return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask

    def build_train_valid_test_datasets(self):
        logging.info('Building T5 datasets.')
        if self._cfg.data.seq_length_dec < self._cfg.data.seq_length * self._cfg.data.masked_lm_prob:
            raise ValueError(
                f"Cannot have decoder max sequence length ({self._cfg.data.seq_length_dec}) less than encoder sequence length ({self._cfg.data.seq_length}) * masked_lm_prob ({self._cfg.data.masked_lm_prob})"
            )
        global_batch_size = self.trainer.world_size * self._cfg.micro_batch_size / self._cfg.tensor_model_parallel_size
        eval_iters = (self.trainer.max_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches
        train_valid_test_num_samples = [
            self.trainer.max_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]
        # Make sure the user specifies dataset type as either 't5' or 't5_prefix_lm' only.
        if self._cfg.data.get('dataset_type', None) is not None:
            if self._cfg.data.get('dataset_type') not in ['t5', 't5_prefix_lm']:
                raise ValueError(
                    f"dataset_type must be either 't5' or 't5_prefix_lm'. found {self._cfg.data.get('dataset_type')}"
                )
        self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
            cfg=self._cfg,
            trainer=self.trainer,
            tokenizer=self.tokenizer,
            data_prefix=self._cfg.data.data_prefix,
            data_impl=self._cfg.data.data_impl,
            splits_string=self._cfg.data.splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            max_seq_length=self._cfg.data.seq_length,
            max_seq_length_dec=self._cfg.data.seq_length_dec,
            masked_lm_prob=self._cfg.data.masked_lm_prob,
            short_seq_prob=self._cfg.data.short_seq_prob,
            seed=self._cfg.seed,
            skip_warmup=self._cfg.data.skip_warmup,
            dataset_type=self._cfg.data.get('dataset_type', 't5'),
            max_ngram_size=self._cfg.get('max_ngram_size', 10),
            mean_ngram_size=self._cfg.get('mean_ngram_size', None),
            geometric_dist=self._cfg.get('geometric_dist', True),
            permutation=self._cfg.get('permutation', False),
            whole_word_masking=self._cfg.get('whole_word_masking', True),
            favor_long_ngrams=self._cfg.get('favor_long_ngrams', False),
        )
        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building T5 datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def list_available_models(self):
        pass
