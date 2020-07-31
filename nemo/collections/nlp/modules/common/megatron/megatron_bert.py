# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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
import random

import numpy as np
import torch
import torch.distributed as dist

from nemo.collections.nlp.modules.common.bert_module import BertModule
from nemo.core.classes import typecheck
from nemo.utils import logging
from nemo.utils.decorators import experimental

if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group

try:
    from megatron.mpu.initialize import set_model_parallel_rank, get_model_parallel_rank, set_model_parallel_world_size
    from megatron.mpu import model_parallel_cuda_manual_seed, get_cuda_rng_tracker
    from megatron.initialize import _set_random_seed, _init_autoresume, initialize_megatron, _initialize_distributed
    from megatron.global_vars import set_global_variables, get_args
    from megatron.model.bert_model import bert_attention_mask_func, bert_extended_attention_mask, bert_position_ids
    from megatron.model.language_model import get_language_model, TransformerLanguageModel
    from megatron.model.utils import init_method_normal, scaled_init_method_normal
except ModuleNotFoundError as err:
    logging.error(f"Could not import {err.name}. Megatron LM is not available. Make sure you are using NeMo on GPUs.")


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

    def __init__(
        self, model_name, config,
    ):

        super().__init__()

        if not os.path.exists(config["vocab_file"]):
            raise ValueError(f'Vocab file not found at {config["vocab_file"]}')

        config['tokenizer_type'] = 'BertWordPieceLowerCase'
        set_global_variables(extra_args_provider=None, args_defaults=config, ignore_unknown_args=True)
        # read arguments back
        args = get_args()


        if args.distributed_backend == 'ddp':
            set_model_parallel_rank(args.rank)
            set_model_parallel_world_size(args.model_parallel_size)
        else:
            _initialize_distributed()

        # Autoresume.
        _init_autoresume()

        # Random seeds for reproducibility.
        if args.rank == 0:
            seed = args.seed
            if seed is not None and seed > 0:
                print('> setting random seeds to {} ...'.format(args.seed))
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

        print(f"MegatronBertEncoder.init: rank={args.rank}, model_parallel_size = {args.model_parallel_size}, dist={dist.is_initialized()}")
            
        self.language_model = TransformerLanguageModel(
            attention_mask_func=bert_attention_mask_func,
            mlp_activation_func=torch.nn.functional.gelu,
            init_method=init_method_normal(args.init_method_std),
            output_layer_init_method=scaled_init_method_normal(args.init_method_std, args.num_layers),
            num_tokentypes=2,
            add_pooler=False,
        )

        # key used for checkpoints.
        self._language_model_key = 'language_model'
        self._hidden_size = self.language_model.hidden_size
        self._rng_initialized = False

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
        if not self._rng_initialized:
            seed = get_args().seed
            offset = seed + 2718
            model_parallel_seed = offset + get_model_parallel_rank()
            get_cuda_rng_tracker().add('model-parallel-rng', model_parallel_seed)

            self._rng_initialized = True

        extended_attention_mask = bert_extended_attention_mask(
            attention_mask, next(self.language_model.parameters()).dtype
        )
        position_ids = bert_position_ids(input_ids)

        sequence_output = self.language_model(
            input_ids, position_ids, extended_attention_mask, tokentype_ids=token_type_ids
        )
        return sequence_output

    def restore_weights(self, restore_path: str):
        """Restores module/model's weights"""
        state_dict = torch.load(restore_path)

        # to load from Megatron pretrained checkpoint
        if 'model' in state_dict:
            self.language_model.load_state_dict(state_dict['model'][self._language_model_key])
        else:
            self.load_state_dict(state_dict)
