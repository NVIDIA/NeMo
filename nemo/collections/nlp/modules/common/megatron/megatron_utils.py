# Copyright 2020 The HuggingFace Inc. team.
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

import json
import os
from typing import List, Optional

import torch
import wget
from transformers import TRANSFORMERS_CACHE, cached_path

from nemo.collections.nlp.modules.common.megatron.megatron_bert import MegatronBertEncoder
from nemo.utils import logging

__all__ = [
    'get_megatron_lm_model',
    'get_megatron_lm_models_list',
    'get_megatron_checkpoint',
    'is_lower_cased_megatron',
    'get_megatron_tokenizer',
]


MEGATRON_CACHE = os.path.join(os.path.dirname(str(TRANSFORMERS_CACHE)), 'megatron')

CONFIGS = {'345m': {"hidden_size": 1024, "num_attention_heads": 16, "num_layers": 24, "max_position_embeddings": 512}}

MEGATRON_CONFIG_MAP = {
    'megatron-bert-345m-uncased': {
        'config': CONFIGS['345m'],
        'checkpoint': 'https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.0/files/release/mp_rank_00/model_optim_rng.pt',
        'vocab': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt',
        'do_lower_case': True,
        'tokenizer_name': 'bert-large-uncased',
    },
    'megatron-bert-345m-cased': {
        'config': CONFIGS['345m'],
        'checkpoint': 'https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/files/release/mp_rank_00/model_optim_rng.pt',
        'vocab': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt',
        'do_lower_case': False,
        'tokenizer_name': 'bert-large-cased',
    },
    'megatron-bert-uncased': {
        'config': None,
        'checkpoint': None,
        'vocab': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt',
        'do_lower_case': True,
        'tokenizer_name': 'bert-large-uncased',
    },
    'megatron-bert-cased': {
        'config': None,
        'checkpoint': None,
        'vocab': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt',
        'do_lower_case': False,
        'tokenizer_name': 'bert-large-cased',
    },
}


def get_megatron_lm_model(
    pretrained_model_name: str,
    config_dict: Optional[dict] = None,
    config_file: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
):
    '''
    Returns the dict of special tokens associated with the model.
    Args:
        pretrained_mode_name: model name from MEGATRON_CONFIG_MAP
            for example: megatron-bert-cased
        config_dict: model configuration parameters
        config_file: path to model configuration file. Takes precedence over config_dict if both supplied.
        checkpoint_file: path to checkpoint file or directory if using model parallel.
    '''
    config = None
    # get default config and checkpoint
    if config_file:
        with open(config_file) as f:
            config = json.load(f)
            # replace dashes with underscores in config keys
            fixed_config = {}
            for key in config.keys():
                fixed_key = key.replace("-", "_")
                if fixed_key == 'max_seq_length':
                    fixed_key = 'max_position_embeddings'
                fixed_config[fixed_key] = config[key]
            # 'vocab_size" no longer used.
            if 'vocab_size' in fixed_config:
                fixed_config.pop('vocab_size')
            config = fixed_config
    elif config_dict:
        config = config_dict
    elif pretrained_model_name in get_megatron_lm_models_list():
        config = get_megatron_config(pretrained_model_name)
    else:
        raise ValueError(f'{pretrained_model_name} is not supported')

    if config is None:
        raise ValueError(f'config_file or config_dict is required for {pretrained_model_name}')

    if not checkpoint_file:
        checkpoint_file = get_megatron_checkpoint(pretrained_model_name)

    vocab = get_megatron_vocab_file(pretrained_model_name)

    # if checkpoint path is a directory, then we automatically compute model parallel size
    if os.path.isdir(checkpoint_file):
        model_parallel_size = len(os.listdir(checkpoint_file))
        logging.info(
            (
                f'restore_path: {checkpoint_file} is a directory. '
                f'Assuming megatron model parallelism with '
                f'model_parallel_size: {model_parallel_size}'
            )
        )
    else:
        model_parallel_size = None

    model = MegatronBertEncoder(
        model_name=pretrained_model_name, config=config, vocab_file=vocab, model_parallel_size=model_parallel_size
    )

    return model, checkpoint_file


def get_megatron_lm_models_list() -> List[str]:
    '''
    Return the list of support Megatron models
    '''
    return list(MEGATRON_CONFIG_MAP.keys())


def get_megatron_config(pretrained_model_name):
    '''
    Returns model config file
    Args:
        pretrained_model_name (str): pretrained model name
    Returns:
        config (dict): contains model configuration: number of hidden layers, number of attention heads, etc
    '''
    return MEGATRON_CONFIG_MAP[pretrained_model_name]['config']


def get_megatron_vocab_file(pretrained_model_name):
    '''
    Gets vocabulary file from cache or downloads it
    Args:
        pretrained_model_name (str): pretrained model name
    Returns:
        path (str): path to the vocab file
    '''
    url = MEGATRON_CONFIG_MAP[pretrained_model_name]['vocab']
    path = cached_path(url, cache_dir=MEGATRON_CACHE)
    return path


def get_megatron_checkpoint(pretrained_model_name):
    '''
    Gets checkpoint file from cache or downloads it
    Args:
        pretrained_model_name (str): pretrained model name
    Returns:
        path (str): path to model checkpoint
    '''
    url = MEGATRON_CONFIG_MAP[pretrained_model_name]['checkpoint']
    if url is None:
        return None

    path = os.path.join(MEGATRON_CACHE, pretrained_model_name)

    if not os.path.exists(path):
        master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if not os.path.exists(path):
            if master_device:
                os.makedirs(MEGATRON_CACHE, exist_ok=True)
                wget.download(url, path)
            # wait until the master process downloads the file and writes it to the cache dir
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    return path


def is_lower_cased_megatron(pretrained_model_name):
    '''
    Returns if the megatron is cased or uncased
    Args:
        pretrained_model_name (str): pretrained model name
    Returns:
        do_lower_cased (bool): whether the model uses lower cased data
    '''
    return MEGATRON_CONFIG_MAP[pretrained_model_name]['do_lower_case']


def get_megatron_tokenizer(pretrained_model_name: str):
    '''
    Takes a pretrained_model_name for megatron such as 'megatron-bert-cased' and returns the according 
    tokenizer name for tokenizer instantiating.
    Args:
        pretrained_model_name: pretrained_model_name for megatron such as 'megatron-bert-cased'
    Returns: 
        tokenizer name for tokenizer instantiating
    '''
    return MEGATRON_CONFIG_MAP[pretrained_model_name]['tokenizer_name']
