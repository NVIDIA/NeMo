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
import os
import tempfile

import onnx
import pytest
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.models import EncDecClassificationModel, EncDecCTCModel, EncDecSpeakerLabelModel
from nemo.collections.asr.modules import ConvASRDecoder, ConvASREncoder


class TestExportable:
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_ConvASREncoder_export_to_onnx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder_instance = ConvASREncoder.from_config_dict(DictConfig(self.encoder_dict)).cuda()
            assert isinstance(encoder_instance, ConvASREncoder)
            filename = os.path.join(tmpdir, 'qn_encoder.onnx')
            encoder_instance.export(output=filename)
            onnx_model = onnx.load(filename)
            onnx.checker.check_model(onnx_model, full_check=True)  # throws when failed
            assert onnx_model.graph.input[0].name == 'audio_signal'
            assert onnx_model.graph.output[0].name == 'outputs'

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_ConvASRDecoder_export_to_onnx(self):
        decoder = ConvASRDecoder.from_config_dict(config=DictConfig(self.decoder_dict)).cuda()
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'qn_decoder.onnx')
            decoder.export(output=filename)
            onnx_model = onnx.load(filename)
            onnx.checker.check_model(onnx_model, full_check=True)  # throws when failed
            assert onnx_model.graph.input[0].name == 'encoder_output'
            assert onnx_model.graph.output[0].name == 'logprobs'

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_EncDecCTCModel_export_to_onnx(self):
        model_config = DictConfig(
            {
                'preprocessor': DictConfig(self.preprocessor),
                'encoder': DictConfig(self.encoder_dict),
                'decoder': DictConfig(self.decoder_dict),
            }
        )
        model = EncDecCTCModel(cfg=model_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'qn.onnx')
            model.export(output=filename)
            onnx_model = onnx.load(filename)
            onnx.checker.check_model(onnx_model, full_check=True)  # throws when failed
            assert onnx_model.graph.input[0].name == 'audio_signal'
            assert onnx_model.graph.output[0].name == 'logprobs'

    def test_EncDecClassificationModel_export_to_onnx(self, speech_classification_model):
        model = speech_classification_model.train()
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'edc.onnx')
            model.export(output=filename)
            onnx_model = onnx.load(filename)
            onnx.checker.check_model(onnx_model, full_check=True)  # throws when failed
            assert onnx_model.graph.input[0].name == 'audio_signal'
            assert onnx_model.graph.output[0].name == 'logits'

    def test_EncDecSpeakerLabelModel_export_to_onnx(self, speaker_label_model):
        model = speaker_label_model.train()
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'sl.onnx')
            model.export(output=filename)
            onnx_model = onnx.load(filename)
            onnx.checker.check_model(onnx_model, full_check=True)  # throws when failed
            assert onnx_model.graph.input[0].name == 'audio_signal'
            assert onnx_model.graph.output[0].name == 'logits'

    def setup_method(self):
        self.preprocessor = {
            'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
            'params': dict({}),
        }

        self.encoder_dict = {
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

        self.decoder_dict = {
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


@pytest.fixture()
def speaker_label_model():
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
    return speaker_model
