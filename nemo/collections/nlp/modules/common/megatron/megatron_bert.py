# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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

import os

import torch
from megatron import get_args, initialize_megatron
from megatron.model import get_language_model
from megatron.model.bert_model import bert_attention_mask_func, bert_extended_attention_mask, bert_position_ids
#from megatron.mpu import get_model_parallel_rank
from megatron.mpu import get_model_parallel_group

from nemo.collections.nlp.modules.common.bert_module import BertModule
from nemo.core.classes import typecheck
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = ['MegatronBertEncoder']


@experimental
class MegatronBertEncoder(BertModule):
    """
    MegatronBERT wraps around the Megatron Language model
    from https://github.com/NVIDIA/Megatron-LM

    Args:
        config_file (str): path to model configuration file.
        vocab_file (str): path to vocabulary file.
        tokenizer_type (str): tokenizer type, currently only 'BertWordPieceLowerCase' supported.
    """

    def __init__(self, model_name, config, vocab_file):

        super().__init__()

        self._model_parallel_size = None
        self._restore_path = None

        if not os.path.exists(vocab_file):
            raise ValueError(f'Vocab file not found at {vocab_file}')

        config["vocab_file"] = vocab_file

        config['tokenizer_type'] = 'BertWordPieceLowerCase'

        config['lazy_mpu_init'] = True

        config['onnx_safe'] = True

        # config['hidden_size'] = 1024 
        # config['num_attention_heads'] = 16 
        # config['num_layers'] = 24
        # config['max_position_embeddings'] = 512

        os.environ["WORLD_SIZE"] = "2"

        def _update_model_parallel_arg(parser):
            parser.set_defaults(model_parallel_size=2)
            return parser
        extra_args_provider = _update_model_parallel_arg

        # Initialize part of Megatron global state that is needed for its constructor.
        # We set 'lazy_mpu_init' flag on to make Megatron do only the initialization that does not depend
        # on ddp be initialized yet (and we don't want Megatron to initialize DDP itself either)
        # and to return a hook for us to call after PTL has torch.distributed initialized
        # (that would be on our first forward() call).
        self._lazy_init_fn = initialize_megatron(
            extra_args_provider=extra_args_provider, args_defaults=config, ignore_unknown_args=True
        )

        # read Megatron arguments back
        args = get_args()

        self.language_model, self._language_model_key = get_language_model(
            attention_mask_func=bert_attention_mask_func, num_tokentypes=2, add_pooler=False
        )

        # key used for checkpoints.
        self._hidden_size = self.language_model.hidden_size

    @property
    def hidden_size(self):
        """
            Property returning hidden size.

            Returns:
                Hidden size.
        """
        return self._hidden_size

    @typecheck()
    def forward(self, input_ids, attention_mask, token_type_ids):
        if self._lazy_init_fn is not None:
            logging.info(f'Finishing megatron mpu init.')
            self._lazy_init_fn()
            # model parallel checkpoints need to be restored after torch.distributed is initialized
            if self._model_parallel_size is not None:
                self.restore_weights(self._restore_path)
            self._lazy_init_fn = None

        extended_attention_mask = bert_extended_attention_mask(
            attention_mask, next(self.language_model.parameters()).dtype
        )
        position_ids = bert_position_ids(input_ids)

        sequence_output = self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=extended_attention_mask,
            tokentype_ids=token_type_ids,
        )
        return sequence_output

    def restore_weights(self, restore_path: str):
        """Restores module/model's weights.
           For model parallel checkpoints the directory structure 
           should be restore_path/mp_rank_0X/model_optim_rng.pt

        Args:
            restore_path (str): restore_path should a file or a directory if using model parallel
        """
        self._restore_path = restore_path
        if os.path.isfile(restore_path):
            logging.info(
                f'restore_path: {restore_path} is a file. Assuming no megatron model parallelism'
            )
            state_dict = torch.load(restore_path)
            # to load from Megatron pretrained checkpoint
            if 'model' in state_dict:
                self.language_model.load_state_dict(state_dict['model'][self._language_model_key])
            else:
                self.load_state_dict(state_dict)
            logging.info(f"weights restored from {restore_path}")
        elif os.path.isdir(restore_path):
            model_parallel_size = len(os.listdir(restore_path))
            self._model_parallel_size = model_parallel_size
            logging.info(
                (
                    f'restore_path: {restore_path} is a directory. '
                    f'Assuming megatron model parallelism with '
                    f'model_parallel_size: {model_parallel_size}'
                )
            )
            # need model parallel groups to restore
            if torch.distributed.is_initialized():
                model_parallel_rank = torch.distributed.get_rank(group=get_model_parallel_group())
                #mp_restore_path = f'{restore_path}/mp_rank_{get_model_parallel_rank():02d}/model_optim_rng.pt'
                mp_restore_path = f'{restore_path}/mp_rank_{model_parallel_rank:02d}/model_optim_rng.pt'
                logging.info(f'Restoring model parallel checkpoint from: {mp_restore_path}')
                state_dict = torch.load(mp_restore_path)
                # to load from Megatron pretrained checkpoint
                if 'model' in state_dict:
                    self.language_model.load_state_dict(state_dict['model'][self._language_model_key])
                else:
                    self.load_state_dict(state_dict)
            else:
                logging.info(f'torch.distributed not initialized yet. Will not restore model parallel checkpoint')
        else:
            logging.error(f'restore_path: {restore_path} must be a file or directory.')


