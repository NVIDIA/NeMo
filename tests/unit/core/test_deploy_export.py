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
from inspect import signature
from collections import OrderedDict
from pathlib import Path
import urllib.request
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

import nemo
import nemo.collections.nlp as nemo_nlp
import nemo.collections.nlp.nm.trainables.common.token_classification_nm
import nemo.collections.tts as nemo_tts

from nemo import logging
from nemo.core import DeploymentFormat as DF
from nemo.core import NeuralModule

# Check if the required libraries and runtimes are installed.
# Only initialize GPU after this runner is activated.
__trt_pycuda_req_satisfied = True
try:
    import pycuda.autoinit

    # This import causes pycuda to automatically manage CUDA context creation and cleanup.
    import pycuda.driver as cuda

    from .tensorrt_loaders import (
        DefaultDataLoader,
        DataLoaderCache,
        OnnxFileLoader,
        OnnxNetworkLoader,
        BuildEngineLoader,
    )
    from .tensorrt_runner import TensorRTRunnerV2
except Exception as e:
    nemo.logging.error('Failed to import: `{}` ({})'.format(str(e), type(e)))
    __trt_pycuda_req_satisfied = False

# create decorator so that tests can be marked with the TRT requirement
requires_trt = pytest.mark.skipif(
    not __trt_pycuda_req_satisfied, reason="TensorRT/PyCuda library required to run test"
)


@pytest.mark.usefixtures("neural_factory")
class TestDeployExport:
    @torch.no_grad()
    def __test_export_route(self, module, out_name, mode, input_example=None):
        # select correct extension based on the output format
        ext = {DF.ONNX: ".onnx", DF.TRTONNX: ".trt.onnx", DF.PYTORCH: ".pt", DF.TORCHSCRIPT: ".ts"}.get(mode, ".onnx")
        out = Path(f"{out_name}{ext}")
        out_name = str(out)

        if out.exists():
            os.remove(out)

        module.eval()
        torch.manual_seed(1)
        deploy_input_example = input_example
        if isinstance(input_example, OrderedDict):
            deploy_input_example = tuple(input_example.values())
            if len(deploy_input_example) == 1:
                deploy_input_example = deploy_input_example[0]
        elif isinstance(input_example, tuple):
            deploy_input_example = input_example if len(input_example) > 1 else input_example[0]

        sig = signature(module.forward)
        pnum = len(sig.parameters)
        outputs_fwd = module.forward(*deploy_input_example) if pnum > 2 else module.forward(deploy_input_example)
        self.nf.deployment_export(
            module=module, output=out_name, input_example=deploy_input_example, d_format=mode, output_example=None,
        )

        assert out.exists() == True

        if mode == DF.TRTONNX:

            data_loader = DefaultDataLoader()
            loader_cache = DataLoaderCache(data_loader)
            profile_shapes = OrderedDict()
            names = list(module.input_ports) + list(module.output_ports)
            names = list(
                filter(
                    lambda x: x
                    not in (module._disabled_deployment_input_ports | module._disabled_deployment_output_ports),
                    names,
                )
            )
            if isinstance(input_example, tuple):
                si = [tuple(input_example[i].shape) for i in range(len(input_example))]
            elif isinstance(input_example, OrderedDict):
                si = [tuple(input_example.values())[i].shape for i in range(len(input_example))]
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

            onnx_loader = OnnxFileLoader(out_name)
            network_loader = OnnxNetworkLoader(onnx_loader, explicit_precision=False)
            model_loader = BuildEngineLoader(
                network_loader,
                max_workspace_size=1 << 30,
                fp16_mode=False,
                int8_mode=False,
                profile_shapes=profile_shapes,
                write_engine=None,
                calibrator=None,
                layerwise=False,
            )

            with TensorRTRunnerV2(model_loader=model_loader) as active_runner:
                input_metadata = active_runner.get_input_metadata()
                if input_metadata is None:
                    logging.critical("For {:}, get_input_metadata() returned None!".format(active_runner.name))
                logging.debug("Runner Inputs: {:}".format(input_metadata))
                feed_dict = loader_cache.load(iteration=0, input_metadata=input_metadata, input_example=input_example)
                inputs = dict()
                input_names = list(input_metadata.keys())
                for i in range(len(input_names)):
                    input_name = input_names[i]
                    if input_name in module._disabled_deployment_input_ports:
                        continue

                    if isinstance(input_example, OrderedDict):
                        for key in input_example.keys():
                            if key in input_name:
                                inputs[input_name] = input_example[key].cpu().numpy()
                    elif isinstance(input_example, tuple):
                        inputs[input_name] = input_example[i].cpu().numpy()
                    else:
                        inputs[input_name] = input_example.cpu().numpy()

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

        elif mode == DF.ONNX:
            # Must recompute because *module* might be different now
            torch.manual_seed(1)
            outputs_fwd = module.forward(*deploy_input_example) if pnum > 2 else module.forward(deploy_input_example)

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            ort_session = ort.InferenceSession(out_name, sess_options, ['CUDAExecutionProvider'])
            print('Execution Providers: ', ort_session.get_providers())
            inputs = dict()
            input_names = (
                list(input_example.keys())
                if isinstance(input_example, OrderedDict)
                else list(module.input_ports.keys())
            )
            ort_inputs = ort_session.get_inputs()

            for node_arg in ort_inputs:
                ort_name = node_arg.name
                for input_name in input_names:
                    if input_name in ort_name or ort_name in input_name:
                        break
                if ort_name not in inputs:
                    inputs[ort_name] = (
                        input_example[input_name].cpu().numpy()
                        if isinstance(input_example, OrderedDict)
                        else input_example.cpu().numpy()
                    )

            output_names = None
            outputs_scr = ort_session.run(output_names, inputs)
            outputs_scr = torch.from_numpy(outputs_scr[0]).cuda()
        elif mode == DF.TORCHSCRIPT:
            tscr = torch.jit.load(out_name)
            torch.manual_seed(1)
            outputs_scr = (
                tscr.forward(*tuple(input_example.values()))
                if isinstance(input_example, OrderedDict)
                else (
                    tscr.forward(*input_example) if isinstance(input_example, tuple) else tscr.forward(input_example)
                )
            )
        elif mode == DF.PYTORCH:
            module.restore_from(out_name)
            torch.manual_seed(1)
            if isinstance(input_example, OrderedDict):
                outputs_scr = module.forward(*tuple(input_example.values()))
            elif isinstance(input_example, tuple) or isinstance(input_example, list):
                outputs_scr = module.forward(*input_example)
            else:
                outputs_scr = module.forward(input_example)

        outputs_scr = (
            outputs_scr[0] if isinstance(outputs_scr, tuple) or isinstance(outputs_scr, list) else outputs_scr
        )
        outputs_fwd = (
            outputs_fwd[0] if isinstance(outputs_fwd, tuple) or isinstance(outputs_fwd, list) else outputs_fwd
        )

        n = outputs_fwd.numel()
        tol = 5.0e-3 if n < 10000 else (5.0e-2 if n < 100000 else (5.0e-1))

        assert (outputs_scr - outputs_fwd).norm(p=2) < tol

        if out.exists():
            os.remove(out)

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.parametrize(
        "input_example, module_name, df_type",
        [
            # TaylorNet export tests.
            (torch.randn(4, 1), "TaylorNet", DF.PYTORCH),
            # TokenClassifier export tests.
            (torch.randn(16, 16, 512), "TokenClassifier", DF.ONNX),
            (torch.randn(16, 16, 512), "TokenClassifier", DF.TORCHSCRIPT),
            (torch.randn(16, 16, 512), "TokenClassifier", DF.PYTORCH),
            pytest.param(torch.randn(16, 16, 512), "TokenClassifier", DF.TRTONNX, marks=requires_trt),
            # JasperDecoderForCTC export tests.
            (torch.randn(34, 1024, 1), "JasperDecoderForCTC", DF.ONNX),
            (torch.randn(34, 1024, 1), "JasperDecoderForCTC", DF.TORCHSCRIPT),
            (torch.randn(34, 1024, 1), "JasperDecoderForCTC", DF.PYTORCH),
            pytest.param(torch.randn(34, 1024, 1), "JasperDecoderForCTC", DF.TRTONNX, marks=requires_trt),
            # JasperEncoder export tests.
            (torch.randn(16, 64, 256), "JasperEncoder", DF.ONNX),
            (torch.randn(16, 64, 256), "JasperEncoder", DF.TORCHSCRIPT),
            (torch.randn(16, 64, 256), "JasperEncoder", DF.PYTORCH),
            pytest.param(torch.randn(16, 64, 256), "JasperEncoder", DF.TRTONNX, marks=requires_trt),
            # QuartznetEncoder export tests.
            (torch.randn(16, 64, 256), "QuartznetEncoder", DF.ONNX),
            (torch.randn(16, 64, 256), "QuartznetEncoder", DF.TORCHSCRIPT),
            (torch.randn(16, 64, 256), "QuartznetEncoder", DF.PYTORCH),
            pytest.param(torch.randn(16, 64, 256), "QuartznetEncoder", DF.TRTONNX, marks=requires_trt),
        ],
    )
    def test_module_export(self, tmpdir, input_example, module_name, df_type):
        """ Tests the module export.

            Args:
                tmpdir: Fixture which will provide a temporary directory.

                input_example: Input to be passed to TaylorNet.

                module_name: Name of the module (section in config file).

                df_type: Parameter denoting type of export to be tested.
        """
        # Create neural module instance.
        module = NeuralModule.import_from_config("tests/configs/test_deploy_export.yaml", module_name)
        # Generate filename in the temporary directory.
        tmp_file_name = str(tmpdir.mkdir("export").join(module_name))
        input_example = input_example.cuda() if input_example is not None else input_example
        # Test export.
        self.__test_export_route(
            module=module, out_name=tmp_file_name, mode=df_type, input_example=input_example,
        )

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.parametrize("df_type", [DF.ONNX, DF.TORCHSCRIPT, DF.PYTORCH])
    def test_hf_bert(self, tmpdir, df_type):
        """ Tests BERT export.

            Args:
                tmpdir: Fixture which will provide a temporary directory.

                df_type: Parameter denoting type of export to be tested.
        """
        bert = nemo.collections.nlp.nm.trainables.common.huggingface.BERT(pretrained_model_name="bert-base-uncased")
        input_example = OrderedDict(
            [
                ("input_ids", torch.randint(low=0, high=16, size=(2, 16)).cuda()),
                ("token_type_ids", torch.randint(low=0, high=2, size=(2, 16)).cuda()),
                ("attention_mask", torch.randint(low=0, high=2, size=(2, 16)).cuda()),
            ]
        )
        # Generate filename in the temporary directory.
        tmp_file_name = str(tmpdir.mkdir("export").join("bert"))
        # Test export.
        self.__test_export_route(module=bert, out_name=tmp_file_name, mode=df_type, input_example=input_example)

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.parametrize("df_type", [DF.TORCHSCRIPT, DF.PYTORCH])
    #
    # TODO WaveGlow.infer uses torch.randn which is required to be seeded
    # for deterministic results. It gets translated to ONNX op like this:
    #
    #   %16020 = RandomNormalLike[dtype = 1](%16019)
    #
    # There is no way to seed it, thus to validate ONNX test flow
    # please use torch.ones
    #
    # @pytest.mark.parametrize("df_type", [DF.ONNX, DF.TORCHSCRIPT, DF.PYTORCH])
    #
    def test_waveglow(self, tmpdir, df_type):
        url = "https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ljspeech/versions/2/files/WaveGlowNM.pt"
        ptfile = "./WaveGlowNM.pt"
        if not Path(ptfile).is_file():
            urllib.request.urlretrieve(url, ptfile)

        module = nemo_tts.WaveGlowInferNM(sample_rate=22050)
        module.restore_from(ptfile)
        module.eval()

        torch.manual_seed(1)
        mel = torch.randn(1, 80, 96).cuda()

        input_example = OrderedDict([("mel_spectrogram", mel)])
        tmp_file_name = str(tmpdir.mkdir("export").join("waveglow"))

        self.__test_export_route(module=module, out_name=tmp_file_name, mode=df_type, input_example=input_example)
