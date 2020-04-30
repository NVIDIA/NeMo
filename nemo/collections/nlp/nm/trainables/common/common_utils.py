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

__all__ = ['get_bert_models_list']

from nemo.collections.nlp.nm.trainables.common.huggingface.huggingface_utils import MODELS

def get_bert_models_list():
    '''
    Return the list of support HuggingFace and Megatron-LM models
    '''
    huggingface_models = []
    for model in MODELS:
        model_names = [x.pretrained_model_name for x in MODELS[model]['class'].list_pretrained_models()]
        huggingface_models.extend(model_names)
    return ['megatron-uncased'] + huggingface_models