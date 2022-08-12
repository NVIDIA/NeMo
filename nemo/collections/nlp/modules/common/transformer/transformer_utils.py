# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


from typing import Optional, Union

from omegaconf.dictconfig import DictConfig

from nemo.collections.common.parts.transformer.transformer_utils import get_megatron_transformer, get_nemo_transformer
from nemo.collections.nlp.modules.common.huggingface.huggingface_decoder import HuggingFaceDecoderModule
from nemo.collections.nlp.modules.common.huggingface.huggingface_encoder import HuggingFaceEncoderModule


def get_huggingface_transformer(
    model_name: Optional[str] = None,
    pretrained: bool = False,
    config_dict: Optional[Union[dict, DictConfig]] = None,
    encoder: bool = True,
) -> Union[HuggingFaceEncoderModule, HuggingFaceDecoderModule]:

    if encoder:
        model = HuggingFaceEncoderModule(model_name, pretrained, config_dict)
    else:
        model = HuggingFaceDecoderModule(model_name, pretrained, config_dict)

    return model
