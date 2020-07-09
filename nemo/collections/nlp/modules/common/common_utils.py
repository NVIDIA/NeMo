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

from typing import List, Optional

from nemo import logging
from nemo.collections.nlp.modules.common.huggingface.huggingface_utils import (
    get_huggingface_lm_model,
    get_huggingface_lm_models_list,
)

__all__ = ['get_pretrained_lm_models_list', 'get_pretrained_lm_model']


def get_pretrained_lm_models_list() -> List[str]:
    '''
    Returns the list of support pretrained models
    '''
    return get_huggingface_lm_models_list()


def get_pretrained_lm_model(
    pretrained_model_name: str, config_file: Optional[str] = None, checkpoint_file: Optional[str] = None
):
    '''
    Returns pretrained model
    Args:
        pretrained_model_name (str): pretrained model name, for example, bert-base-uncased.
            See the full list by calling get_pretrained_lm_models_list()
        config_file (str): path to the model configuration file
        checkpoint_file (str): path to the pretrained model checkpoint
    Returns:
        Pretrained model (NM)
    '''
    if pretrained_model_name in get_huggingface_lm_models_list():
        model = get_huggingface_lm_model(config_file=config_file, pretrained_model_name=pretrained_model_name)
    else:
        raise ValueError(f'{pretrained_model_name} is not supported')

    if checkpoint_file:
        model.restore_from(checkpoint_file)
        logging.info(f"{pretrained_model_name} model restored from {checkpoint_file}")
    return model
