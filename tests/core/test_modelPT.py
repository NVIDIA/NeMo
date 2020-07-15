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
from unittest import TestCase
from omegaconf import DictConfig
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelConfig
import pytest


class ModelPTSaveRestore(TestCase):
    @pytest.mark.unit
    def test_save_restore(self):
        preprocessor = {
                    'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
                    'params': dict({})
            }

        encoder = {
                'cls': 'nemo.collections.asr.modules.ConvASREncoder',
                'params': {'feat_in': 64,
                           'activation': 'relu',
                           'conv_mask': True,
                           'jasper': [{'filters': 1024,
                                       'repeat': 1,
                                       'kernel': [1],
                                       'stride': [1],
                                       'dilation': [1],
                                       'dropout': 0.0,
                                       'residual': False,
                                       'separable': True,
                                       'se': True,
                                       'se_context_size': -1}]}
        }

        decoder = {
            'cls': 'nemo.collections.asr.modules.ConvASRDecoder',
            'params': {'feat_in': 1024,
                       'num_classes': 28,
                       'vocabulary': [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'"]}
        }

        modelConfig = EncDecCTCModelConfig(preprocessor=DictConfig(preprocessor),
                                           encoder=DictConfig(encoder),
                                           decoder=DictConfig(decoder))
        modelConfig.pl = None
        asr_model = EncDecCTCModel(cfg=modelConfig)
        asr_model.train()
        print(modelConfig)
        asr_model.save_to(save_path="here.nemo")
