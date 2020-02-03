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
from pathlib import Path

import torch
from ruamel.yaml import YAML

import nemo
import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp
import nemo.collections.nlp.nm.trainables.common.token_classification_nm
from tests.common_setup import NeMoUnitTest


class TestDeployExport(NeMoUnitTest):
    def setUp(self):
        """ Setups neural factory so it will use GPU instead of CPU. """
        NeMoUnitTest.setUp(self)

        # Perform computations on GPU.
        self.nf._placement = nemo.core.DeviceType.GPU

    def __test_export_route(self, module, out_name, mode, input_example=None):
        out = Path(out_name)
        if out.exists():
            os.remove(out)

        self.nf.deployment_export(module=module, output=out_name, input_example=input_example, d_format=mode)

        self.assertTrue(out.exists())
        if out.exists():
            os.remove(out)

    def test_simple_module_export(self):
        simplest_module = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        self.__test_export_route(
            module=simplest_module,
            out_name="simple.pt",
            mode=nemo.core.DeploymentFormat.TORCHSCRIPT,
            input_example=None,
        )

    def test_TokenClassifier_module_export(self):
        t_class = nemo.collections.nlp.nm.trainables.common.token_classification_nm.TokenClassifier(
            hidden_size=512, num_classes=16, use_transformer_pretrained=False
        )
        self.__test_export_route(
            module=t_class,
            out_name="t_class.pt",
            mode=nemo.core.DeploymentFormat.TORCHSCRIPT,
            input_example=torch.randn(16, 16, 512).cuda(),
        )

    def test_TokenClassifier_module_onnx_export(self):
        t_class = nemo.collections.nlp.nm.trainables.common.token_classification_nm.TokenClassifier(
            hidden_size=512, num_classes=16, use_transformer_pretrained=False
        )
        self.__test_export_route(
            module=t_class,
            out_name="t_class.onnx",
            mode=nemo.core.DeploymentFormat.ONNX,
            input_example=torch.randn(16, 16, 512).cuda(),
        )

    def test_jasper_decoder_export_ts(self):
        j_decoder = nemo_asr.JasperDecoderForCTC(feat_in=1024, num_classes=33)
        self.__test_export_route(
            module=j_decoder, out_name="j_decoder.ts", mode=nemo.core.DeploymentFormat.TORCHSCRIPT, input_example=None
        )

    def test_hf_bert_ts(self):
        bert = nemo.collections.nlp.nm.trainables.common.huggingface.BERT(pretrained_model_name="bert-base-uncased")
        input_example = (
            torch.randint(low=0, high=16, size=(2, 16)).cuda(),
            torch.randint(low=0, high=1, size=(2, 16)).cuda(),
            torch.randint(low=0, high=1, size=(2, 16)).cuda(),
        )
        self.__test_export_route(
            module=bert, out_name="bert.ts", mode=nemo.core.DeploymentFormat.TORCHSCRIPT, input_example=input_example
        )

    def test_hf_bert_pt(self):
        bert = nemo.collections.nlp.nm.trainables.common.huggingface.BERT(pretrained_model_name="bert-base-uncased")
        self.__test_export_route(module=bert, out_name="bert.pt", mode=nemo.core.DeploymentFormat.PYTORCH)

    def test_jasper_encoder_to_onnx(self):
        with open("tests/data/jasper_smaller.yaml") as file:
            yaml = YAML(typ="safe")
            jasper_model_definition = yaml.load(file)

        jasper_encoder = nemo_asr.JasperEncoder(
            conv_mask=False,
            feat_in=jasper_model_definition['AudioToMelSpectrogramPreprocessor']['features'],
            **jasper_model_definition['JasperEncoder']
        )

        self.__test_export_route(
            module=jasper_encoder,
            out_name="jasper_encoder.onnx",
            mode=nemo.core.DeploymentFormat.ONNX,
            input_example=(torch.randn(16, 64, 256).cuda(), torch.randn(256).cuda()),
        )
