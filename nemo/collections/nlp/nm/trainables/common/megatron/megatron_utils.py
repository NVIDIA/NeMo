# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
# Copyright 2020 The HuggingFace Inc. team.
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
# =============================================================================

import os

import torch
import wget
from transformers import TRANSFORMERS_CACHE, cached_path

__all__ = [
    'MEGATRON_CACHE',
    'MEGATRON_CONFIG_MAP',
    'CONFIGS',
    'get_megatron_lm_models_list',
    'get_megatron_config_file',
    'get_megatron_vocab_file',
    'get_megatron_checkpoint',
]

MEGATRON_CACHE = os.path.join(os.path.dirname(str(TRANSFORMERS_CACHE)), 'megatron')

CONFIGS = {'345m': {"hidden-size": 1024, "num-attention-heads": 16, "num-layers": 24, "max-seq-length": 512}}

MEGATRON_CONFIG_MAP = {
    'megatron-bert-345m-uncased': {
        'config': CONFIGS['345m'],
        'checkpoint': 'https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.0/files/release/mp_rank_00/model_optim_rng.pt',
        'vocab': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt',
        'do_lower_case': True,
    },
    'megatron-bert-uncased': {
        'config': None,
        'checkpoint': None,
        'vocab': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt',
        'do_lower_case': True,
    },
    'megatron-bert-cased': {
        'config': None,
        'vocab': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt',
        'do_lower_case': False,
    },
}


def get_megatron_lm_models_list():
    '''
    Return the list of support Megatron models
    '''
    return list(MEGATRON_CONFIG_MAP.keys())


def get_megatron_config_file(pretrained_model_name):
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
    path = os.path.join(MEGATRON_CACHE, pretrained_model_name)

    if not os.path.exists(path):
        master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if not os.path.exists(path):
            if master_device:
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
