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
import os

import pytest
from omegaconf import DictConfig

from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE


@pytest.fixture()
def asr_model(test_data_dir):
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
        'params': {'feat_in': 1024, 'num_classes': -1, 'vocabulary': None},
    }

    tokenizer = {'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128"), 'type': 'wpe'}

    modelConfig = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'tokenizer': DictConfig(tokenizer),
        }
    )

    model_instance = EncDecCTCModelBPE(cfg=modelConfig)
    return model_instance


class TestEncDecCTCModel:
    @pytest.mark.unit
    def test_constructor(self, asr_model):
        asr_model.train()
        # TODO: make proper config and assert correct number of weights
        # Check to/from config_dict:
        confdict = asr_model.to_config_dict()
        instance2 = EncDecCTCModelBPE.from_config_dict(confdict)
        assert isinstance(instance2, EncDecCTCModelBPE)

    @pytest.mark.unit
    def test_save_restore_artifact(self, asr_model):
        asr_model.train()
        asr_model.save_to('./ctc_bpe.nemo')

        new_model = EncDecCTCModelBPE.restore_from('./ctc_bpe.nemo')
        assert isinstance(new_model, type(asr_model))
        assert new_model.vocab_path == 'vocab.txt'

        assert len(new_model.tokenizer.tokenizer.get_vocab()) == 128

        if os.path.exists('./ctc_bpe.nemo'):
            os.remove('./ctc_bpe.nemo')
