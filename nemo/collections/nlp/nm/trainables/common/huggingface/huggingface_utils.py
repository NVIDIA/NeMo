# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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

from nemo.collections.nlp.nm.trainables.common.huggingface.albert_nm import Albert
from nemo.collections.nlp.nm.trainables.common.huggingface.bert_nm import BERT
from nemo.collections.nlp.nm.trainables.common.huggingface.roberta_nm import Roberta

__all__ = ['MODELS', 'get_huggingface_lm_model', 'get_huggingface_lm_models_list']


def get_huggingface_lm_model(pretrained_model_name, bert_config=None):
    '''
    Returns the dict of special tokens associated with the model.
    Args:
    pretrained_mode_name ('str'): name of the pretrained model from the hugging face list,
        for example: bert-base-cased
    bert_config: path to model configuration file.
    '''
    model_type = pretrained_model_name.split('-')[0]
    if model_type in MODELS:
        if bert_config:
            return MODELS[model_type]['class'](config_filename=bert_config)
        else:
            return MODELS[model_type]['class'](pretrained_model_name=pretrained_model_name)
    else:
        raise ValueError(f'{pretrained_model_name} is not supported')


MODELS = {
    'bert': {'default': 'bert-base-uncased', 'class': BERT},
    'roberta': {'default': 'roberta-base', 'class': Roberta},
    'albert': {'default': 'albert-base-v2', 'class': Albert},
}


def get_huggingface_lm_models_list():
    '''
    Returns the list of supported HuggingFace models
    '''
    huggingface_models = []
    for model in MODELS:
        model_names = [x.pretrained_model_name for x in MODELS[model]['class'].list_pretrained_models()]
        huggingface_models.extend(model_names)
    return huggingface_models
