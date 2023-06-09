# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from omegaconf import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import build_train_valid_test_datasets
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model

from nemo.utils import logging

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ['MegatronGPTSFTModel']


class MegatronUGPTModel(MegatronGPTModel):
    """
    Megatron GPT with a FIM/UL2 loss. References:

    1. UL2 - https://arxiv.org/abs/2205.05131
    2. U-PaLM - https://arxiv.org/abs/2210.11399
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self._add_special_tokens_to_tokenizer()
        self._resize_model_embeddings()
        self._maybe_resize_output_layer()

    def _add_special_tokens_to_tokenizer(self):
        MegatronT5Model.add_special_tokens_to_tokenizer(
            tokenizer=self.tokenizer,
            tokenizer_cfg=self.cfg.tokenizer,
            dataset_type="ul2",
            add_sentinel_tokens_in_reverse_order=self._cfg.tokenizer.get(
                "add_sentinel_tokens_in_reverse_order", False
            ),
            add_sentinel_tokens_first=self._cfg.tokenizer.get("add_sentinel_tokens_first", False),
            add_base_tokens=True,
        )
        # NOTE: This should only happen for the GPT2 tokenizer.
        if self.tokenizer.pad_id is None:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.sentinel_tokens = self._get_sentinel_token_ids()

    def _get_sentinel_token_ids(self):
        """Returns all the sentinel token ids in a list.
        Sentinel tokens include tokenizer.additional_special_token_ids and IDs already present in the tokenizer like <extra_id_0>, ... ,<extra_id_999>
        Sentinel tokens also exclude UL2 tokens if they are present in the tokenizer.
        """
        sentinel_tokens = set()
        # The additional_special_token_ids already exclude bos, eos, pad etc.
        for token_id in self.tokenizer.additional_special_tokens_ids:
            # Exclude UL2 tokens.
            if self.tokenizer.ids_to_tokens([token_id])[0] in ['<extra_id_r>', '<extra_id_s>', '<extra_id_x>']:
                continue
            else:
                sentinel_tokens.add(token_id)

        # Try and add <extra_id_xx> tokens that may already be in the tokenizer vocab.
        for i in range(self.cfg.tokenizer.get('num_sentinel_tokens', 0)):
            token = f"<extra_id_{i}>"
            token_ids = self.tokenizer.tokens_to_ids(token)
            if isinstance(token_ids, list) and len(token_ids) > 1:
                continue
            token_id = token_ids if isinstance(token_ids, int) else token_ids[0]
            if token_id not in sentinel_tokens:
                sentinel_tokens.add(token_id)
        return sorted(list(sentinel_tokens))

    def _resize_model_embeddings(self):
        # Resize the model embedding layer.
        self._model_embeddings_resized = False
        num_added_tokens = len(self.tokenizer.vocab) - self.model.word_embeddings_weight().size(0)
        if num_added_tokens > 0:
            logging.info(f"Resizing the model's embedding layer by adding {num_added_tokens} tokens.")
            with torch.no_grad():
                mean = self.model.word_embeddings_weight().mean().item()
                std = self.model.word_embeddings_weight().std().item()
                new_embeddings = self.model.word_embeddings_weight().new_empty(
                    len(self.tokenizer.vocab), self.model.word_embeddings_weight().size(1)
                )
                new_embeddings.normal_(mean=mean, std=std)
                new_embeddings[:-num_added_tokens] = self.model.word_embeddings_weight()
                self.model.word_embeddings_weight().set_(new_embeddings)
                # Broadcast the embeddings from rank 0 to all other embedding ranks.
                # torch.distributed.all_reduce(
                #     self.model.word_embeddings_weight().data,
                #     group=parallel_state.get_embedding_group(),
                #     op=torch.distributed.ReduceOp.AVG,
                # )

                self._model_embeddings_resized = True

    def _resize_model_embeddings_broadcast(self):
        if self._model_embeddings_resized:
            # Broadcast the embeddings from rank 0 to all other embedding ranks.
            torch.distributed.all_reduce(
                self.model.word_embeddings_weight().data,
                group=parallel_state.get_embedding_group(),
                op=torch.distributed.ReduceOp.AVG,
            )

    def _maybe_resize_output_layer(self):
        # Maybe resize the output layer if using untied embeddings and output weights.
        self._output_layer_resized = False

        # Attribute for sentinal
        if not self.cfg.get('share_embeddings_and_output_weights', True):
            # Resize the model embedding layer.
            if self.cfg.megatron_amp_O2:
                num_added_tokens = len(
                    self.tokenizer.vocab
                ) - self.model.module.language_model.output_layer.weight.size(0)
                output_layer_weight = self.model.module.language_model.output_layer.weight
            else:
                num_added_tokens = len(self.tokenizer.vocab) - self.model.language_model.output_layer.weight.size(0)
                output_layer_weight = self.model.language_model.output_layer.weight
            logging.info(f"Resizing the model's output layer by adding {num_added_tokens} tokens.")
            if num_added_tokens > 0:
                with torch.no_grad():
                    mean = output_layer_weight.mean().item()
                    std = output_layer_weight.std().item()
                    new_output_layer = output_layer_weight.new_empty(
                        len(self.tokenizer.vocab), output_layer_weight.size(1)
                    )
                    new_output_layer.normal_(mean=mean, std=std)
                    new_output_layer[:-num_added_tokens] = output_layer_weight
                    # TODO: Fix this later.
                    # Issue: restore from a basic GPT model and continue training.
                    # Dont deal with initial noise - just copy the weights.
                    # 0..999 tokens already exist - copy the last N tokens duplicated.

                    # Todo
                    # O2 vs O1 - .module missing
                    # VP will list of models, only last have some decoder
                    # Property for self.model (in main)
                    # guerantee - called something.output_layer

                    # change state of current model
                    # update logic here to deal with decoder update

                    self.model.module.language_model.output_layer.weight.set_(new_output_layer)
                    # Broadcast the embeddings from rank 0 to all other embedding ranks.
                    '''
                    torch.distributed.all_reduce(
                        output_layer_weight.data, group=parallel_state.get_embedding_group(),
                        op=torch.distributed.ReduceOp.AVG
                    )
                    '''

                    self._output_layer_resized = True

    def _maybe_resize_output_layer_broadcast(self):
        if self._output_layer_resized:
            # TODO: Cleanup before merge
            pass
            # if self.cfg.megatron_amp_O2:
            #     output_layer_weight = self.model.module.language_model.output_layer.weight
            # else:
            #     output_layer_weight = self.model.language_model.output_layer.weight
            # # Broadcast the embeddings from rank 0 to all other embedding ranks.
            # torch.distributed.all_reduce(
            #     output_layer_weight.data, group=parallel_state.get_embedding_group(),
            #     op=torch.distributed.ReduceOp.AVG
            # )

    def setup(self, stage=None):
        super().setup(stage)
        # Resize the model embedding layer.
        self._resize_model_embeddings_broadcast()
        # Maybe resize the output layer if using untied embeddings and output weights.
        self._maybe_resize_output_layer_broadcast()

    def build_train_valid_test_datasets(self):
        logging.info('Building U-GPT datasets.')
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")
        global_batch_size = self.cfg.global_batch_size
        max_train_steps = self.trainer.max_steps
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches

        train_valid_test_num_samples = [
            max_train_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]

        if self.trainer.limit_val_batches <= 1.0 and isinstance(self.trainer.limit_val_batches, float):
            train_valid_test_num_samples[
                1
            ] = 1  # This is to make sure we only have one epoch on every validation iteration

        self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
            cfg=self.cfg,
            trainer=self.trainer,
            tokenizer=self.tokenizer,
            data_prefix=self.cfg.data.data_prefix,
            data_impl=self.cfg.data.data_impl,
            splits_string=self.cfg.data.splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            max_seq_length=self.cfg.data.seq_length,
            masked_lm_prob=self.cfg.data.masked_lm_prob,
            short_seq_prob=self.cfg.data.short_seq_prob,
            seed=self.cfg.seed,
            skip_warmup=self.cfg.data.skip_warmup,
            dataset_type='u_gpt',
            max_ngram_size=self.cfg.data.get('max_ngram_size', 10),
            mean_ngram_size=self.cfg.data.get('mean_ngram_size', None),
            geometric_dist=self.cfg.data.get('geometric_dist', True),
            permutation=self.cfg.data.get('permutation', False),
            whole_word_masking=self.cfg.data.get('whole_word_masking', True),
            favor_long_ngrams=self.cfg.data.get('favor_long_ngrams', False),
            respect_document_boundaries=self.cfg.data.get('respect_document_boundaries', False),
            data_impl_kwargs=self.cfg.data.get('data_impl_kwargs', {}),
            sentinel_tokens=self.sentinel_tokens,
        )
        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building U-GPT datasets.')

        return self._train_ds, self._validation_ds, self._test_ds
