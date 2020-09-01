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
    HUGGINGFACE_MODELS
)
from nemo.collections.nlp.modules.common.megatron.megatron_utils import (
    get_megatron_lm_model,
    get_megatron_lm_models_list,
)

__all__ = ['get_pretrained_lm_models_list', 'get_pretrained_lm_model']


def get_pretrained_lm_models_list() -> List[str]:
    """
    Returns the list of support pretrained models
    """
    return get_megatron_lm_models_list() + get_huggingface_pretrained_lm_models_list()

def get_lm_models_list() -> List[str]:
    """
    Returns the list of support models
    """
    return ["megatron"] + list(HUGGINGFACE_MODELS.keys())


def get_pretrained_lm_model(
    model_type: str,
    pretrained_model_name: Optional[str] = None,
    config_dict: Optional[dict] = None,
    config_file: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
) -> BertModule:
    """
    Returns pretrained model

    Args:
        model_type: model type, e.g. bert, megatron, etc.
        pretrained_model_name: pretrained model name, for example, bert-base-uncased.
            See the full list by calling get_pretrained_lm_models_list()
        config_dict: path to the model configuration dictionary
        config_file: path to the model configuration file
        checkpoint_file: path to the pretrained model checkpoint

    Returns:
        Pretrained BertModule
    """


    # check valid model type    
    if model_type not in get_lm_models_list():
        raise ValueError(f'model_type needs to be from {get_lm_models_list()}, however got {model_type}')

    # warning when user passes both configuration dict and file
    if config_dict and config_file:
        logging.warning(f"Both config_dict and config_file were found, defaulting to use config_file: {config_file} will be used.")

    # check either config or pretrained_model name is specified, not both
    if (config_dict or config_file) and pretrained_model_name:
        raise ValueError(f"Either specify model XOR pretrained_mode_name, but got both")

    # check valid optional pretrained_model_name
    if pretrained_model_name and pretrained_model_name not in get_pretrained_lm_models_list():
        raise ValueError(f'pretrained_mode_name needs to be from {get_pretrained_lm_models_list()}, however got {pretrained_mode_name}')



    if model_type in HUGGINGFACE_MODELS.keys():
        model = get_huggingface_lm_model(
            model_type=model_type, config_dict=config_dict, config_file=config_file, pretrained_model_name=pretrained_model_name
        )
    elif model_type == "megatron":
        if pretrained_model_name in get_megatron_lm_models_list():
            model, default_checkpoint_file = get_megatron_lm_model(
                config_dict=config_dict,
                config_file=config_file,
                pretrained_model_name=pretrained_model_name,
                checkpoint_file=checkpoint_file,
            )
        else:
            raise ValueError(f'{pretrained_model_name} is not supported')

    if checkpoint_file:
        model.restore_weights(restore_path=checkpoint_file)

    return model
