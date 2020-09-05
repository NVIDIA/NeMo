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

from typing import List, Optional

from nemo.collections.nlp.modules.common.bert_module import BertModule
from nemo.collections.nlp.modules.common.huggingface.huggingface_utils import (
    get_huggingface_lm_model,
    get_huggingface_pretrained_lm_models_list,
)
from nemo.collections.nlp.modules.common.megatron.megatron_utils import (
    get_megatron_lm_model,
    get_megatron_lm_models_list,
)
from nemo.utils import logging

__all__ = ['get_pretrained_lm_models_list', 'get_lm_model']


def get_pretrained_lm_models_list() -> List[str]:
    """
    Returns the list of supported pretrained model names
    """
    return get_megatron_lm_models_list() + get_huggingface_pretrained_lm_models_list()


def get_lm_model(
    pretrained_model_name: str,
    config_dict: Optional[dict] = None,
    config_file: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
) -> BertModule:
    """
    Helper function to instantiate a language model encoder, either from scratch or a pretrained model.
    If only pretrained_model_name are passed, a pretrained model is returned.
    If a configuration is passed, whether as a file or dictionary, the model is initialized with random weights.

    Args:
        pretrained_model_name: pretrained model name, for example, bert-base-uncased or megatron-bert-cased.
            See get_pretrained_lm_models_list() for full list.
        config_dict: path to the model configuration dictionary
        config_file: path to the model configuration file
        checkpoint_file: path to the pretrained model checkpoint

    Returns:
        Pretrained BertModule
    """

    # check valid model type
    if not pretrained_model_name or pretrained_model_name not in get_pretrained_lm_models_list():
        raise ValueError(
            f'pretrained_model_name needs to be from {get_pretrained_lm_models_list()}, however got {pretrained_model_name}'
        )

    # warning when user passes both configuration dict and file
    if config_dict and config_file:
        logging.warning(
            f"Both config_dict and config_file were found, defaulting to use config_file: {config_file} will be used."
        )

    if pretrained_model_name in get_huggingface_pretrained_lm_models_list():
        model = get_huggingface_lm_model(
            config_dict=config_dict, config_file=config_file, pretrained_model_name=pretrained_model_name,
        )
    elif "megatron" in pretrained_model_name:
        model, checkpoint_file = get_megatron_lm_model(
            config_dict=config_dict,
            config_file=config_file,
            pretrained_model_name=pretrained_model_name,
            checkpoint_file=checkpoint_file,
        )

    if checkpoint_file:
        model.restore_weights(restore_path=checkpoint_file)

    return model
