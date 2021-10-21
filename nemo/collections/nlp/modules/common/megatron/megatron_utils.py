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

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import wget
from torch.hub import _get_torch_home

from nemo.collections.nlp.modules.common.megatron.megatron_bert import MegatronBertEncoder
from nemo.utils import AppState, logging

__all__ = [
    "get_megatron_lm_model",
    "get_megatron_lm_models_list",
    "get_megatron_checkpoint",
    "is_lower_cased_megatron",
    "get_megatron_tokenizer",
]


torch_home = _get_torch_home()

if not isinstance(torch_home, str):
    logging.info("Torch home not found, caching megatron in cwd")
    torch_home = os.getcwd()

MEGATRON_CACHE = os.path.join(torch_home, "megatron")


CONFIGS = {"345m": {"hidden_size": 1024, "num_attention_heads": 16, "num_layers": 24, "max_position_embeddings": 512}}

MEGATRON_CONFIG_MAP = {
    "megatron-gpt-345m": {
        "config": CONFIGS["345m"],
        "checkpoint": "models/nvidia/megatron_lm_345m/versions/v0.0/files/release/mp_rank_00/model_optim_rng.pt",
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
        "merges_file": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
        "do_lower_case": False,
        "tokenizer_name": "gpt2",
    },
    "megatron-bert-345m-uncased": {
        "config": CONFIGS["345m"],
        "checkpoint": "https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.0/files/release/mp_rank_00/model_optim_rng.pt",
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
        "do_lower_case": True,
        "tokenizer_name": "bert-large-uncased",
    },
    "megatron-bert-345m-cased": {
        "config": CONFIGS["345m"],
        "checkpoint": "https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/files/release/mp_rank_00/model_optim_rng.pt",
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
        "do_lower_case": False,
        "tokenizer_name": "bert-large-cased",
    },
    "megatron-bert-uncased": {
        "config": None,
        "checkpoint": None,
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
        "do_lower_case": True,
        "tokenizer_name": "bert-large-uncased",
    },
    "megatron-bert-cased": {
        "config": None,
        "checkpoint": None,
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
        "do_lower_case": False,
        "tokenizer_name": "bert-large-cased",
    },
    "biomegatron-bert-345m-uncased": {
        "config": CONFIGS["345m"],
        "checkpoint": "https://api.ngc.nvidia.com/v2/models/nvidia/biomegatron345muncased/versions/0/files/MegatronBERT.pt",
        "vocab": "https://api.ngc.nvidia.com/v2/models/nvidia/biomegatron345muncased/versions/0/files/vocab.txt",
        "do_lower_case": True,
        "tokenizer_name": "bert-large-uncased",
    },
    "biomegatron-bert-345m-cased": {
        "config": CONFIGS["345m"],
        "checkpoint": "https://api.ngc.nvidia.com/v2/models/nvidia/biomegatron345mcased/versions/0/files/MegatronBERT.pt",
        "vocab": "https://api.ngc.nvidia.com/v2/models/nvidia/biomegatron345mcased/versions/0/files/vocab.txt",
        "do_lower_case": False,
        "tokenizer_name": "bert-large-cased",
    },
}


def get_megatron_lm_model(
    pretrained_model_name: str,
    config_dict: Optional[dict] = None,
    config_file: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
    vocab_file: Optional[str] = None,
    merges_file: Optional[str] = None,
) -> Tuple[MegatronBertEncoder, str]:
    """
    Returns MegatronBertEncoder and a default or user specified path to the checkpoint file

    Args:
        pretrained_mode_name: model name from MEGATRON_CONFIG_MAP
            for example: megatron-bert-cased
        config_dict: model configuration parameters
        config_file: path to model configuration file. Takes precedence over config_dict if both supplied.
        checkpoint_file: path to checkpoint file or directory if using model parallel.
        vocab_file: path to vocab file

    Returns:
        model: MegatronBertEncoder
        checkpoint_file: path to checkpoint file or directory
    """
    raise ValueError(
        f'megatron-lm bert has been deprecated in NeMo 1.5. Please use an earlier release of NeMo. Megatron bert support will be added back to NeMo in a future release.'
    )
    # config = None
    # # get default config and checkpoint
    # if config_file:
    #     with open(config_file) as f:
    #         config = json.load(f)
    #         # replace dashes with underscores in config keys
    #         fixed_config = {}
    #         for key in config.keys():
    #             fixed_key = key.replace("-", "_")
    #             if fixed_key == 'max_seq_length':
    #                 fixed_key = 'max_position_embeddings'
    #             fixed_config[fixed_key] = config[key]
    #         # 'vocab_size" no longer used.
    #         if 'vocab_size' in fixed_config:
    #             fixed_config.pop('vocab_size')
    #         config = fixed_config
    # elif config_dict:
    #     config = config_dict
    # elif pretrained_model_name in get_megatron_lm_models_list():
    #     config = get_megatron_config(pretrained_model_name)
    # else:
    #     raise ValueError(f"{pretrained_model_name} is not supported")

    # if config is None:
    #     raise ValueError(f"config_file or config_dict is required for {pretrained_model_name}")

    # if not checkpoint_file:
    #     checkpoint_file = get_megatron_checkpoint(pretrained_model_name)

    # if not vocab_file:
    #     vocab_file = get_megatron_vocab_file(pretrained_model_name)

    # if not merges_file:
    #     merges_file = get_megatron_merges_file(pretrained_model_name)

    # app_state = AppState()
    # if app_state.model_parallel_size is not None and app_state.model_parallel_rank is not None:
    #     # model parallel already known from .nemo restore
    #     model_parallel_size = app_state.model_parallel_size
    #     model_parallel_rank = app_state.model_parallel_rank
    # elif os.path.isdir(checkpoint_file):
    #     # starting training from megatron-lm checkpoint
    #     mp_ranks = glob.glob(os.path.join(checkpoint_file, 'mp_rank*'))
    #     model_parallel_size = len(mp_ranks)
    #     app_state.model_parallel_size = model_parallel_size
    #     logging.info(
    #         (
    #             f'restore_path: {checkpoint_file} is a directory. '
    #             f'Assuming megatron model parallelism with '
    #             f'model_parallel_size: {model_parallel_size}'
    #         )
    #     )
    #     # try to get local rank from global
    #     local_rank = None
    #     try:
    #         local_rank = int(os.environ['LOCAL_RANK'])
    #     except:
    #         logging.info('Global variable LOCAL_RANK not yet specified')
    #     if local_rank is not None:
    #         app_state.local_rank = local_rank
    #     else:
    #         # if local is None then we are on the main process
    #         local_rank = 0
    #     model_parallel_rank = compute_model_parallel_rank(local_rank, model_parallel_size)
    #     app_state.model_parallel_rank = model_parallel_rank
    # else:
    #     model_parallel_size = None
    #     model_parallel_rank = None

    # model = MegatronBertEncoder(
    #     model_name=pretrained_model_name,
    #     config=config,
    #     vocab_file=vocab_file,
    #     model_parallel_size=model_parallel_size,
    #     model_parallel_rank=model_parallel_rank,
    # )

    # return model, checkpoint_file


def compute_model_parallel_rank(local_rank, model_parallel_size):
    return local_rank % model_parallel_size


def get_megatron_lm_models_list() -> List[str]:
    """
    Returns the list of supported Megatron-LM models
    """
    return list(MEGATRON_CONFIG_MAP.keys())


def get_megatron_config(pretrained_model_name: str) -> Dict[str, int]:
    """
    Returns Megatron-LM model config file

    Args:
        pretrained_model_name (str): pretrained model name

    Returns:
        config (dict): contains model configuration: number of hidden layers, number of attention heads, etc
    """
    _check_megatron_name(pretrained_model_name)
    return MEGATRON_CONFIG_MAP[pretrained_model_name]["config"]


def _check_megatron_name(pretrained_model_name: str) -> None:
    megatron_model_list = get_megatron_lm_models_list()
    if pretrained_model_name not in megatron_model_list:
        raise ValueError(f'For Megatron-LM models, choose from the following list: {megatron_model_list}')


def get_megatron_vocab_file(pretrained_model_name: str) -> str:
    """
    Gets vocabulary file from cache or downloads it

    Args:
        pretrained_model_name: pretrained model name

    Returns:
        path: path to the vocab file
    """
    _check_megatron_name(pretrained_model_name)
    url = MEGATRON_CONFIG_MAP[pretrained_model_name]["vocab"]

    path = os.path.join(MEGATRON_CACHE, pretrained_model_name + "_vocab")
    path = _download(path, url)
    return path


def get_megatron_merges_file(pretrained_model_name: str) -> str:
    """
    Gets merge file from cache or downloads it

    Args:
        pretrained_model_name: pretrained model name

    Returns:
        path: path to the vocab file
    """
    if 'gpt' not in pretrained_model_name.lower():
        return None
    _check_megatron_name(pretrained_model_name)
    url = MEGATRON_CONFIG_MAP[pretrained_model_name]["merges_file"]

    path = os.path.join(MEGATRON_CACHE, pretrained_model_name + "_merges")
    path = _download(path, url)
    return path


def get_megatron_checkpoint(pretrained_model_name: str) -> str:
    """
    Gets checkpoint file from cache or downloads it
    Args:
        pretrained_model_name: pretrained model name
    Returns:
        path: path to model checkpoint
    """
    _check_megatron_name(pretrained_model_name)
    url = MEGATRON_CONFIG_MAP[pretrained_model_name]["checkpoint"]
    path = os.path.join(MEGATRON_CACHE, pretrained_model_name)
    return _download(path, url)


def _download(path: str, url: str):
    """
    Gets a file from cache or downloads it

    Args:
        path: path to the file in cache
        url: url to the file
    Returns:
        path: path to the file in cache
    """
    if url is None:
        return None

    if not os.path.exists(path):
        master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if not os.path.exists(path):
            if master_device:
                os.makedirs(MEGATRON_CACHE, exist_ok=True)
                logging.info(f"Downloading from {url}")
                wget.download(url, path)
            # wait until the master process downloads the file and writes it to the cache dir
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    return path


def is_lower_cased_megatron(pretrained_model_name):
    """
    Returns if the megatron is cased or uncased

    Args:
        pretrained_model_name (str): pretrained model name
    Returns:
        do_lower_cased (bool): whether the model uses lower cased data
    """
    _check_megatron_name(pretrained_model_name)
    return MEGATRON_CONFIG_MAP[pretrained_model_name]["do_lower_case"]


def get_megatron_tokenizer(pretrained_model_name: str):
    """
    Takes a pretrained_model_name for megatron such as "megatron-bert-cased" and returns the according 
    tokenizer name for tokenizer instantiating.

    Args:
        pretrained_model_name: pretrained_model_name for megatron such as "megatron-bert-cased"
    Returns: 
        tokenizer name for tokenizer instantiating
    """
    _check_megatron_name(pretrained_model_name)
    return MEGATRON_CONFIG_MAP[pretrained_model_name]["tokenizer_name"]
