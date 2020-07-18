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

import tempfile

import numpy as np
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

    return EncDecCTCModel(cfg=modelConfig)


class TestFileIO:
    @pytest.mark.unit
    def test_to_from_config_file(self, asr_model):
        """" Test makes sure that the second instance created with the same configuration (BUT NOT checkpoint)
        has different weights. """

        with tempfile.NamedTemporaryFile() as fp:
            yaml_filename = fp.name
            asr_model.to_config_file(path2yaml_file=yaml_filename)
            next_instance = EncDecCTCModel.from_config_file(path2yaml_file=yaml_filename)

            assert isinstance(next_instance, EncDecCTCModel)

            assert len(next_instance.decoder.vocabulary) == 28
            assert asr_model.num_weights == next_instance.num_weights

            w1 = asr_model.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()
            w2 = next_instance.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()

            assert np.array_equal(w1, w2) == False

    @pytest.mark.unit
    @pytest.mark.pleasefixme
    def test_save_restore_from_nemo_file(self, asr_model):
        """" Test makes sure that the second instance created from the same configuration AND checkpoint 
        has the same weights. """

        with tempfile.NamedTemporaryFile() as fp:
            filename = fp.name
            asr_model.save_to(save_path=filename)

            # Restore!
            asr_model2 = EncDecCTCModel.restore_from(restore_path=filename)

            assert len(asr_model.decoder.vocabulary) == len(asr_model2.decoder.vocabulary)
            assert asr_model.num_weights == asr_model2.num_weights

            w1 = asr_model.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()
            w2 = asr_model2.encoder.encoder[0].mconv[0].conv.weight.data.detach().cpu().numpy()
            assert np.array_equal(w1, w2)
