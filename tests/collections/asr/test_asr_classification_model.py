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

import pytest
from omegaconf import DictConfig

from nemo.collections.asr.models import EncDecClassificationModel


class TestEncDecClassificationModel:
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
                        'filters': 32,
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
            'cls': 'nemo.collections.asr.modules.ConvASRDecoderClassification',
            'params': {'feat_in': 32, 'num_classes': 30,},
        }

        modelConfig = DictConfig(
            {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder)}
        )
        asr_model = EncDecClassificationModel(cfg=modelConfig)
        asr_model.train()

        conv_cnt = (64 * 32 * 1 + 32) + (64 * 1 * 1 + 32)  # separable kernel + bias + pointwise kernel + bias
        bn_cnt = (4 * 32) * 2  # 2 * moving averages
        dec_cnt = 32 * 30 + 30  # fc + bias

        param_count = conv_cnt + bn_cnt + dec_cnt
        assert asr_model.num_weights == param_count

        # Check to/from config_dict:
        confdict = asr_model.to_config_dict()
        instance2 = EncDecClassificationModel.from_config_dict(confdict)

        assert isinstance(instance2, EncDecClassificationModel)
