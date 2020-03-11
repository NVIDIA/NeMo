# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2020 NVIDIA. All Rights Reserved.
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

import copy
import os
import sys
from collections import OrderedDict
from os import path, sys
from pathlib import Path
from unittest import TestCase

import numpy as np
# git clone git@github.com:microsoft/onnxruntime.git
# cd onnxruntime
#
# ./build.sh --update --build --config RelWithDebInfo  --build_shared_lib --parallel \
#     --cudnn_home /usr/lib/x86_64-linux-gnu --cuda_home /usr/local/cuda \
#     --tensorrt_home .../TensorRT --use_tensorrt --enable_pybind --build_wheel
#
# pip install --upgrade ./build/Linux/RelWithDebInfo/dist/*.whl
import onnxruntime as ort
import pytest
import torch
from ruamel.yaml import YAML

import nemo
import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp
import nemo.collections.nlp.nm.trainables.common.token_classification_nm
from nemo import logging

# Check if the required libraries and runtimes are installed.
try:
    # Only initialize GPU after this runner is activated.
    import pycuda.autoinit

    # This import causes pycuda to automatically manage CUDA context creation and cleanup.
    import pycuda.driver as cuda

    sys.path.append(os.path.join(path.dirname(path.abspath(__file__)), 'trt_ONNX'))
    import tensorrt_loaders as loaders
    import tensorrt_runner as runner
except:
    # Skip tests.
    pytestmark = pytest.mark.skip


@pytest.mark.usefixtures("neural_factory")
class TestDeployExport(TestCase):
    def __test_export_route(self, module, out_name, mode, input_example=None):
        out = Path(out_name)
        if out.exists():
            os.remove(out)

        outputs_fwd = (
            (module.forward(*input_example) if isinstance(input_example, tuple) else module.forward(input_example))
            if input_example is not None
            else None
        )
        self.nf.deployment_export(
            module=module, output=out_name, input_example=input_example, d_format=mode, output_example=outputs_fwd
        )

        tol = 5.0e-3
        out = Path(out_name)
        self.assertTrue(out.exists())

        if mode == nemo.core.DeploymentFormat.TRTONNX:

            data_loader = loaders.DefaultDataLoader()
            loader_cache = loaders.DataLoaderCache(data_loader)
            profile_shapes = OrderedDict()
            names = list(module.input_ports) + list(module.output_ports)

            if isinstance(input_example, tuple):
                si = [tuple(input_example[i].shape) for i in range(len(input_example))]
            else:
                si = [tuple(input_example.shape)]
            if isinstance(outputs_fwd, tuple):
                fi = [tuple(outputs_fwd[i].shape) for i in range(len(outputs_fwd))]
            else:
                fi = [tuple(outputs_fwd.shape)]
            si = si + fi
            i = 0
            for name in names:
                profile_shapes[name] = [si[i]] * 3
                i = i + 1

            onnx_loader = loaders.OnnxFileLoader(out_name)
            network_loader = loaders.OnnxNetworkLoader(onnx_loader, explicit_precision=False)
            model_loader = loaders.BuildEngineLoader(
                network_loader,
                max_workspace_size=1 << 30,
                fp16_mode=False,
                int8_mode=False,
                profile_shapes=profile_shapes,
                write_engine=None,
                calibrator=None,
                layerwise=False,
            )

            with runner.TensorRTRunnerV2(model_loader=model_loader) as active_runner:
                input_metadata = active_runner.get_input_metadata()
                if input_metadata is None:
                    logging.critical("For {:}, get_input_metadata() returned None!".format(active_runner.name))
                logging.debug("Runner Inputs: {:}".format(input_metadata))
                feed_dict = loader_cache.load(iteration=0, input_metadata=input_metadata, input_example=input_example)
                inputs = dict()
                input_names = list(module.input_ports)
                for i in range(len(input_names)):
                    input_name = (
                        "encoded_lengths"
                        if type(module).__name__ == "JasperEncoder" and input_names[i] == "length"
                        else input_names[i]
                    )
                    inputs[input_name] = (
                        input_example[i].cpu().numpy()
                        if isinstance(input_example, tuple)
                        else input_example.cpu().numpy()
                    )

                out_dict = active_runner.infer(feed_dict=feed_dict, output=outputs_fwd)
                for ov in out_dict.values():
                    outputs_scr = torch.from_numpy(ov).cuda()
                    break

                outputs = []
                outputs.append(copy.deepcopy(out_dict))
                logging.debug(
                    "Received outputs: {:}".format(
                        ["{:}: {:}".format(name, out.shape) for name, out in out_dict.items()]
                    )
                )
                logging.info("Output Buffers: {:}".format(outputs))

            inpex = []
            for ie in feed_dict.values():  # loader_cache.cache[0].values():
                if ie.dtype.type is np.int32:
                    inpex.append(torch.from_numpy(ie).long().cuda())
                else:
                    inpex.append(torch.from_numpy(ie).cuda())
                if len(inpex) == len(input_example):
                    break
            inpex = tuple(inpex)
            outputs_fwd = module.forward(*inpex)

        elif mode == nemo.core.DeploymentFormat.ONNX:
            # Must recompute because *module* might be different now
            outputs_fwd = (
                module.forward(*input_example) if isinstance(input_example, tuple) else module.forward(input_example)
            )
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            ort_session = ort.InferenceSession(out_name, sess_options, ['CUDAExecutionProvider'])
            print('Execution Providers: ', ort_session.get_providers())
            inputs = dict()
            input_names = list(module.input_ports)
            for i in range(len(input_names)):
                input_name = (
                    "encoded_lengths"
                    if type(module).__name__ == "JasperEncoder" and input_names[i] == "length"
                    else input_names[i]
                )
                inputs[input_name] = (
                    input_example[i].cpu().numpy() if isinstance(input_example, tuple) else input_example.cpu().numpy()
                )
            outputs_scr = ort_session.run(None, inputs)
            outputs_scr = torch.from_numpy(outputs_scr[0]).cuda()
        elif mode == nemo.core.DeploymentFormat.TORCHSCRIPT:
            scr = torch.jit.load(out_name)
            if isinstance(module, nemo.backends.pytorch.tutorials.TaylorNet):
                input_example = torch.randn(4, 1).cuda()
                outputs_fwd = module.forward(input_example)
            outputs_scr = (
                scr.forward(*input_example) if isinstance(input_example, tuple) else scr.forward(input_example)
            )
        elif mode == nemo.core.DeploymentFormat.PYTORCH:
            module.load_state_dict(torch.load(out_name))
            module.eval()
            outputs_scr = (
                module.forward(*input_example) if isinstance(input_example, tuple) else module.forward(input_example)
            )

        outputs_scr = (
            outputs_scr[0] if isinstance(outputs_scr, tuple) or isinstance(outputs_scr, list) else outputs_scr
        )
        outputs_fwd = (
            outputs_fwd[0] if isinstance(outputs_fwd, tuple) or isinstance(outputs_fwd, list) else outputs_fwd
        )

        self.assertLess((outputs_scr - outputs_fwd).norm(p=2), tol)

        if out.exists():
            os.remove(out)

    def __test_export_route_all(self, module, out_name, input_example=None):
        if input_example is not None:
            self.__test_export_route(
                module, out_name + '.trt.onnx', nemo.core.DeploymentFormat.TRTONNX, input_example=input_example
            )
            self.__test_export_route(module, out_name + '.onnx', nemo.core.DeploymentFormat.ONNX, input_example)
            self.__test_export_route(module, out_name + '.pt', nemo.core.DeploymentFormat.PYTORCH, input_example)
        self.__test_export_route(module, out_name + '.ts', nemo.core.DeploymentFormat.TORCHSCRIPT, input_example)

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_simple_module_export(self):
        simplest_module = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        self.__test_export_route_all(
            module=simplest_module, out_name="simple", input_example=None,
        )

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_TokenClassifier_module_export(self):
        t_class = nemo.collections.nlp.nm.trainables.common.token_classification_nm.TokenClassifier(
            hidden_size=512, num_classes=16, use_transformer_pretrained=False
        )
        self.__test_export_route_all(
            module=t_class, out_name="t_class", input_example=torch.randn(16, 16, 512).cuda(),
        )

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_jasper_decoder(self):
        j_decoder = nemo_asr.JasperDecoderForCTC(feat_in=1024, num_classes=33)
        self.__test_export_route_all(
            module=j_decoder, out_name="j_decoder", input_example=torch.randn(34, 1024, 1).cuda(),
        )

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_hf_bert(self):
        bert = nemo.collections.nlp.nm.trainables.common.huggingface.BERT(pretrained_model_name="bert-base-uncased")
        input_example = (
            torch.randint(low=0, high=16, size=(2, 16)).cuda(),
            torch.randint(low=0, high=2, size=(2, 16)).cuda(),
            torch.randint(low=0, high=2, size=(2, 16)).cuda(),
        )
        self.__test_export_route_all(module=bert, out_name="bert", input_example=input_example)

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_jasper_encoder(self):
        with open("tests/data/jasper_smaller.yaml") as file:
            yaml = YAML(typ="safe")
            jasper_model_definition = yaml.load(file)

        jasper_encoder = nemo_asr.JasperEncoder(
            conv_mask=False,
            feat_in=jasper_model_definition['AudioToMelSpectrogramPreprocessor']['features'],
            **jasper_model_definition['JasperEncoder']
        )

        self.__test_export_route_all(
            module=jasper_encoder,
            out_name="jasper_encoder",
            input_example=(torch.randn(1, 64, 64).cuda(), torch.randint(low=0, high=2, size=(2, 16)).cuda()),
        )

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_quartz_encoder(self):
        with open("tests/data/quartznet_test.yaml") as file:
            yaml = YAML(typ="safe")
            quartz_model_definition = yaml.load(file)

        jasper_encoder = nemo_asr.JasperEncoder(
            feat_in=quartz_model_definition['AudioToMelSpectrogramPreprocessor']['features'],
            **quartz_model_definition['JasperEncoder']
        )

        self.__test_export_route_all(
            module=jasper_encoder,
            out_name="quartz_encoder",
            input_example=(torch.randn(1, 64, 256).cuda(), torch.randint(1, (16,)).cuda()),
        )
