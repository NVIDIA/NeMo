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

import pytest
from omegaconf import DictConfig

from nemo.collections.asr.modules import ConvASRDecoder, ConvASREncoder
from nemo.collections.nlp.modules.common import (
    BertEncoder,
    BertPretrainingTokenClassifier,
    MegatronBertEncoder,
    TokenClassifier,
    get_pretrained_lm_model,
)


class TestExportable:
    @pytest.mark.unit
    def test_ConvASREncoder_export_to_onnx(self):
        encoder_dict = {
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
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder_instance = ConvASREncoder.from_config_dict(DictConfig(encoder_dict))
            assert isinstance(encoder_instance, ConvASREncoder)
            filename = os.path.join(tmpdir, 'qn_encoder.onnx')
            encoder_instance.export(output=filename)

    @pytest.mark.unit
    def test_token_classifier_export_to_onnx(self):
        for num_layers in [1, 2, 4]:
            tclassifier = TokenClassifier(hidden_size=256, num_classes=16, num_layers=num_layers)
            with tempfile.TemporaryDirectory() as tmpdir:
                filename = os.path.join(tmpdir, 'tclassifier.onnx')
                tclassifier.export(output=filename)

    @pytest.mark.unit
    def test_bert_pretraining_export_to_onnx(self):
        for num_layers in [1, 2, 4]:
            bptclassifier = BertPretrainingTokenClassifier(hidden_size=256, num_classes=16, num_layers=num_layers)
            with tempfile.TemporaryDirectory() as tmpdir:
                filename = os.path.join(tmpdir, 'bptclassifier.onnx')
                bptclassifier.export(output=filename)

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_hf_bert(self):
        """ Tests BERT Encoder export.

            Args:
                tmpdir: Fixture which will provide a temporary directory.
        """
        bert = get_pretrained_lm_model('bert-base-uncased')
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate filename in the temporary directory.
            tmp_file_name = os.path.join(tmpdir, "hf_bert.onnx")
            # Test export.
            bert.export(tmp_file_name)

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_megatron_lm(self):
        """ Tests Megatron export.

            Args:
                tmpdir: Fixture which will provide a temporary directory.

                df_type: Parameter denoting type of export to be tested.
        """
        model_name = "megatron-bert-345m-uncased"
        megatron = get_pretrained_lm_model(model_name)
        with tempfile.TemporaryDirectory() as tmpdir:

            # Generate filename in the temporary directory.
            tmp_file_name = os.path.join("megatron-bert.onnx")
            # Test export.
            megatron.export(tmp_file_name)

    @pytest.mark.unit
    def test_convasr_decoder_export_to_onnx(self):
        decoder_params = {
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
        decoder = ConvASRDecoder.from_config_dict(config=DictConfig(decoder_params))
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'qn_decoder.onnx')
            decoder.export(output=filename)
