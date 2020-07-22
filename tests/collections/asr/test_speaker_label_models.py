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

from nemo.collections.asr.models import EncDecSpeakerLabelModel


class EncDecSpeechLabelModelTest(TestCase):
    @pytest.mark.unit
    def test_constructor(self):
        preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}
        encoder = {
            'cls': 'nemo.collections.asr.modules.ConvASREncoder',
            'params': {
                'feat_in': 64,
                'activation': 'relu',
                'conv_mask': True,
                'jasper': [
                    {
                        'filters': 512,
                        'repeat': 1,
                        'kernel': [1],
                        'stride': [1],
                        'dilation': [1],
                        'dropout': 0.0,
                        'residual': False,
                        'separable': False,
                    }
                ],
            },
        }

        decoder = {
            'cls': 'nemo.collections.asr.modules.SpeakerDecoder',
            'params': {'feat_in': 512, 'num_classes': 2, 'pool_mode': 'xvector', 'emb_sizes': [1024]},
        }

        modelConfig = DictConfig(
            {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder)}
        )
        speaker_model = EncDecSpeakerLabelModel(cfg=modelConfig)
        speaker_model.train()
        # TODO: make proper config and assert correct number of weights

        # Check to/from config_dict:
        confdict = speaker_model.to_config_dict()
        instance2 = EncDecSpeakerLabelModel.from_config_dict(confdict)
        self.assertTrue(isinstance(instance2, EncDecSpeakerLabelModel))
