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
from omegaconf import DictConfig

from nemo.collections.asr.models import EncDecCTCModel


@pytest.fixture()
def asr_model():
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

    model_instance = EncDecCTCModel(cfg=modelConfig)
    return model_instance


class TestEncDecCTCModel:
    @pytest.mark.unit
    def test_constructor(self, asr_model):
        asr_model.train()
        # TODO: make proper config and assert correct number of weights
        # Check to/from config_dict:
        confdict = asr_model.to_config_dict()
        instance2 = EncDecCTCModel.from_config_dict(confdict)
        assert isinstance(instance2, EncDecCTCModel)

    @pytest.mark.unit
    def test_vocab_change(self, asr_model):
        old_vocab = copy.deepcopy(asr_model.decoder.vocabulary)
        nw1 = asr_model.num_weights
        asr_model.change_vocabulary(new_vocabulary=old_vocab)
        # No change
        assert nw1 == asr_model.num_weights
        new_vocab = copy.deepcopy(old_vocab)
        new_vocab.append('!')
        new_vocab.append('$')
        new_vocab.append('@')
        asr_model.change_vocabulary(new_vocabulary=new_vocab)
        # fully connected + bias
        assert asr_model.num_weights == nw1 + 3 * (asr_model.decoder._feat_in + 1)
