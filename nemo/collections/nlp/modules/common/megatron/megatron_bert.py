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
from apex.transformer.enums import AttnMaskType
from apex.transformer.parallel_state import (
    get_model_parallel_group,
    model_parallel_is_initialized,
    set_pipeline_model_parallel_rank,
    set_pipeline_model_parallel_world_size,
)
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.modules.common.bert_module import BertModule
from nemo.collections.nlp.modules.common.megatron.language_model import get_language_model
from nemo.core.classes import typecheck
from nemo.utils import logging
from nemo.utils.app_state import AppState

__all__ = ['MegatronBertEncoder']


class MegatronBertEncoder(BertModule):
    """
    MegatronBERT wraps around the Megatron Language model
    from https://github.com/NVIDIA/Megatron-LM

    Args:
        config_file (str): path to model configuration file.
        vocab_file (str): path to vocabulary file.
        tokenizer_type (str): tokenizer type, currently only 'BertWordPieceLowerCase' supported.
    """

    def __init__(self, model_name, config, vocab_file, model_parallel_size=None, model_parallel_rank=None):

        super().__init__()

        raise ValueError(
            f'megatron-lm bert has been deprecated in NeMo 1.5. Please use an earlier release of NeMo. Megatron bert support will be added back to NeMo in a future release.'
        )

        # self._model_parallel_size = model_parallel_size
        # self._model_parallel_rank = model_parallel_rank
        # self._restore_path = None
        # self._app_state = None
        # self._model_name = model_name

        # if 'vocab_size' in config:
        #     self._vocab_size = config.pop('vocab_size')
        # else:
        #     self._vocab_size = None

        # self._hidden_size = config.get('hidden_size')

        # if not os.path.exists(vocab_file):
        #     raise ValueError(f'Vocab file not found at {vocab_file}')

        # # convert config to dictionary
        # if isinstance(config, DictConfig):
        #     config = OmegaConf.to_container(config)
        # config["vocab_file"] = vocab_file
        # config['tokenizer_type'] = 'BertWordPieceLowerCase'
        # config['lazy_mpu_init'] = True
        # config['onnx_safe'] = True

        # num_tokentypes = config.pop('num_tokentypes', 2)

        # # configure globals for megatron
        # set_pipeline_model_parallel_rank(0)  # pipeline model parallelism not implemented in NeMo
        # set_pipeline_model_parallel_world_size(1)  # pipeline model parallelism not implemented in NeMo

        # self.language_model, self._language_model_key = get_language_model(
        #     encoder_attn_mask_type=AttnMaskType.padding, num_tokentypes=num_tokentypes, add_pooler=False
        # )

        # self.config = OmegaConf.create(config)
        # # key used for checkpoints
        # self._hidden_size = self.language_model.hidden_size

    @property
    def hidden_size(self):
        """
        Property returning hidden size.

        Returns:
            Hidden size.
        """
        return self._hidden_size

    @property
    def vocab_size(self):
        """
        Property returning vocab size.

        Returns:
            vocab size.
        """
        return self._vocab_size

    @typecheck()
    def forward(self, input_ids, attention_mask, token_type_ids=None):

        extended_attention_mask = bert_extended_attention_mask(attention_mask)
        position_ids = bert_position_ids(input_ids)

        sequence_output = self.language_model(
            enc_input_ids=input_ids,
            enc_position_ids=position_ids,
            enc_attn_mask=extended_attention_mask,
            tokentype_ids=token_type_ids,
        )
        return sequence_output

    def _load_checkpoint(self, filename):
        """Helper function for loading megatron checkpoints.

        Args:
            filename (str): Path to megatron checkpoint.
        """
        state_dict = torch.load(filename, map_location='cpu')
        if 'checkpoint_version' in state_dict:
            if state_dict['checkpoint_version'] is not None:
                set_megatron_checkpoint_version(state_dict['checkpoint_version'])
                logging.info(
                    f"Megatron-lm checkpoint version found. Setting checkpoint_version to {state_dict['checkpoint_version']}."
                )
        else:
            logging.warning('Megatron-lm checkpoint version not found. Setting checkpoint_version to 0.')
            set_megatron_checkpoint_version(0)
        # to load from Megatron pretrained checkpoint
        if 'model' in state_dict:
            self.language_model.load_state_dict(state_dict['model'][self._language_model_key])
        else:
            self.load_state_dict(state_dict)

        logging.info(f"Checkpoint loaded from from {filename}")

    def restore_weights(self, restore_path: str):
        """Restores module/model's weights.
           For model parallel checkpoints the directory structure
           should be restore_path/mp_rank_0X/model_optim_rng.pt

        Args:
            restore_path (str): restore_path should a file or a directory if using model parallel
        """
        self._restore_path = restore_path

        if os.path.isfile(restore_path):
            self._load_checkpoint(restore_path)
        elif os.path.isdir(restore_path):
            # need model parallel groups to restore model parallel checkpoints
            if model_parallel_is_initialized():
                model_parallel_rank = torch.distributed.get_rank(group=get_model_parallel_group())
                mp_restore_path = f'{restore_path}/mp_rank_{model_parallel_rank:02d}/model_optim_rng.pt'
                self._load_checkpoint(mp_restore_path)
            else:
                logging.info(f'torch.distributed not initialized yet. Will not restore model parallel checkpoint')
        else:
            logging.error(f'restore_path: {restore_path} must be a file or directory.')


def get_megatron_checkpoint_version():
    app_state = AppState()
    return app_state._megatron_checkpoint_version


def set_megatron_checkpoint_version(version: int = None):
    app_state = AppState()
    app_state._megatron_checkpoint_version = version


def bert_extended_attention_mask(attention_mask):
    # We create a 3D attention mask from a 2D tensor mask.
    # [b, 1, s]
    attention_mask_b1s = attention_mask.unsqueeze(1)
    # [b, s, 1]
    attention_mask_bs1 = attention_mask.unsqueeze(2)
    # [b, s, s]
    attention_mask_bss = attention_mask_b1s * attention_mask_bs1
    # [b, 1, s, s]
    extended_attention_mask = attention_mask_bss.unsqueeze(1)

    # Convert attention mask to binary:
    extended_attention_mask = extended_attention_mask < 0.5

    return extended_attention_mask


def bert_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    return position_ids
