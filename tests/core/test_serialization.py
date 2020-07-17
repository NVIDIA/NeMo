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

import pytest
from omegaconf import DictConfig

from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.modules import SpectrogramAugmentation
from nemo.core.classes.common import Serialization


class SerializationTest(TestCase):
    @pytest.mark.unit
    def test_from_config_dict_with_cls(self):
        """Here we test that instantiation works for configs with cls class path in them.
        Note that just Serialization.from_config_dict can be used to create an object"""
        config = DictConfig(
            {
                'cls': 'nemo.collections.asr.modules.SpectrogramAugmentation',
                'params': {'rect_freq': 50, 'rect_masks': 5, 'rect_time': 120,},
            }
        )
        obj = Serialization.from_config_dict(config=config)
        self.assertTrue(isinstance(obj, SpectrogramAugmentation))

    def test_from_config_dict_without_cls(self):
        """Here we test that instantiation works for configs without cls class path in them.
        IMPORTANT: in this case, correct class type should call from_config_dict. This should work for Models."""
        preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}
        encoder = {
            'cls': 'nemo.collections.asr.modules.ConvASREncoder',
            'params': {
                'feat_in': 64,
                'activation': 'relu',
                'conv_mask': True,
                'jasper': [
                    {
                        'filters': 1024,
                        'repeat': 1,
                        'kernel': [1],
                        'stride': [1],
                        'dilation': [1],
                        'dropout': 0.0,
                        'residual': False,
                        'separable': True,
                        'se': True,
                        'se_context_size': -1,
                    }
                ],
            },
        }

        decoder = {
            'cls': 'nemo.collections.asr.modules.ConvASRDecoder',
            'params': {
                'feat_in': 1024,
                'num_classes': 28,
                'vocabulary': [
                    ' ',
                    'a',
                    'b',
                    'c',
                    'd',
                    'e',
                    'f',
                    'g',
                    'h',
                    'i',
                    'j',
                    'k',
                    'l',
                    'm',
                    'n',
                    'o',
                    'p',
                    'q',
                    'r',
                    's',
                    't',
                    'u',
                    'v',
                    'w',
                    'x',
                    'y',
                    'z',
                    "'",
                ],
            },
        }
        modelConfig = DictConfig(
            {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder)}
        )
        obj = EncDecCTCModel.from_config_dict(config=modelConfig)
        self.assertTrue(isinstance(obj, EncDecCTCModel))
