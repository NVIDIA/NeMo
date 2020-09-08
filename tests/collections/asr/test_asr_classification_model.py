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

import copy

import pytest
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.models import EncDecClassificationModel


@pytest.fixture()
def speech_classification_model():
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
        {
            'preprocessor': DictConfig(preprocessor),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'labels': ListConfig(["dummy_cls_{}".format(i + 1) for i in range(30)]),
        }
    )
    model = EncDecClassificationModel(cfg=modelConfig)
    return model


class TestEncDecClassificationModel:
    @pytest.mark.unit
    def test_constructor(self, speech_classification_model):
        asr_model = speech_classification_model.train()

        conv_cnt = (64 * 32 * 1 + 32) + (64 * 1 * 1 + 32)  # separable kernel + bias + pointwise kernel + bias
        bn_cnt = (4 * 32) * 2  # 2 * moving averages
        dec_cnt = 32 * 30 + 30  # fc + bias

        param_count = conv_cnt + bn_cnt + dec_cnt
        assert asr_model.num_weights == param_count

        # Check to/from config_dict:
        confdict = asr_model.to_config_dict()
        instance2 = EncDecClassificationModel.from_config_dict(confdict)

        assert isinstance(instance2, EncDecClassificationModel)

    @pytest.mark.unit
    def test_vocab_change(self, speech_classification_model):
        asr_model = speech_classification_model.train()

        old_labels = copy.deepcopy(asr_model._cfg.labels)
        nw1 = asr_model.num_weights
        asr_model.change_labels(new_labels=old_labels)
        # No change
        assert nw1 == asr_model.num_weights
        new_labels = copy.deepcopy(old_labels)
        new_labels.append('dummy_cls_31')
        new_labels.append('dummy_cls_32')
        new_labels.append('dummy_cls_33')
        asr_model.change_labels(new_labels=new_labels)
        # fully connected + bias
        assert asr_model.num_weights == nw1 + 3 * (asr_model.decoder._feat_in + 1)
