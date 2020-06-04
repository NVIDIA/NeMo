# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2019 NVIDIA. All Rights Reserved.
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
# =============================================================================

import os

import pytest
from ruamel.yaml import YAML

from nemo.collections.asr.models import ASRConvCTCModel


@pytest.mark.usefixtures("neural_factory")
class TestNeMoModels:
    @pytest.mark.unit
    def test_quartznet_creation(self):
        yaml = YAML(typ="safe")
        with open(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../examples/asr/configs/jasper_an4.yaml"))
        ) as file:
            model_definition = yaml.load(file)
        model = ASRConvCTCModel(
            preprocessor_params=model_definition['AudioToMelSpectrogramPreprocessor'],
            encoder_params=model_definition['JasperEncoder'],
            decoder_params=model_definition['JasperDecoderForCTC'],
        )
        assert model.num_weights > 0
        assert len(model.modules) == 3

    @pytest.mark.unit
    def test_quartznet_nemo_file_export_and_import(self, tmpdir):
        yaml = YAML(typ="safe")
        with open(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../examples/asr/configs/jasper_an4.yaml"))
        ) as file:
            model_definition = yaml.load(file)
        model = ASRConvCTCModel(
            preprocessor_params=model_definition['AudioToMelSpectrogramPreprocessor'],
            encoder_params=model_definition['JasperEncoder'],
            decoder_params=model_definition['JasperDecoderForCTC'],
        )
        nemo_file = str(tmpdir.mkdir("tmp_export_import").join("deleteme.nemo"))
        model.save_to(nemo_file)
        assert os.path.exists(nemo_file)
        new_qn = ASRConvCTCModel.from_pretrained(model_info=nemo_file)
        assert model.num_weights == new_qn.num_weights
