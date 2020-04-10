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

from unittest import TestCase
import os
import pytest
from nemo.collections.asr.models import QuartzNet
from nemo.backends.pytorch.tutorials import MSELoss, RealFunctionDataLayer, TaylorNet
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    AudioSignal,
    AxisKind,
    AxisKindAbstract,
    AxisType,
    ChannelType,
    ElementType,
    MelSpectrogramType,
    MFCCSpectrogramType,
    NeuralPortNmTensorMismatchError,
    NeuralType,
    NeuralTypeComparisonResult,
    SpectrogramType,
    VoidType,
)
from ruamel.yaml import YAML

@pytest.mark.usefixtures("neural_factory")
class NeMoModelsTests(TestCase):
    @pytest.mark.unit
    def test_quartznet_creation(self):
        yaml = YAML(typ="safe")
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../examples/asr/configs/jasper_an4.yaml"))) as file:
            model_definition = yaml.load(file)
        model = QuartzNet(preprocessor_params=model_definition['AudioToMelSpectrogramPreprocessor'],
                          encoder_params=model_definition['JasperEncoder'],
                          decoder_params=model_definition['JasperDecoderForCTC'])
        self.assertTrue(model.num_weights > 0)
        self.assertEqual(len(model.modules), 3)