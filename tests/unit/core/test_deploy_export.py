# ! /usr/bin/python

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
import librosa
import types
from collections import OrderedDict
from pathlib import Path
import urllib.request
import numpy as np

# git clone git@github.com:microsoft/onnxruntime.git
# cd onnxruntime
#
# -*- coding: utf-8 -*-
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
import nemo.collections.tts as nemo_tts
# from nemo.collections.tts.parts.waveglow import WaveGlow

from glow import load_and_setup_model
# import nemo.collections.tts.parts.waveglow as glow

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


def convert_conv_1d_to_2d(conv1d):
    conv2d = torch.nn.Conv2d(conv1d.weight.size(1),
                             conv1d.weight.size(0),
                             (conv1d.weight.size(2), 1),
                             stride=(conv1d.stride[0], 1),
                             dilation=(conv1d.dilation[0], 1),
                             padding=(conv1d.padding[0], 0))
    conv2d.weight.data[:, :, :, 0] = conv1d.weight.data
    conv2d.bias.data = conv1d.bias.data
    return conv2d

def convert_WN_1d_to_2d_(WN):
    """
    Modifies the WaveNet like affine coupling layer in-place to use 2-d convolutions
    """
    WN.start = convert_conv_1d_to_2d(WN.start)
    WN.end = convert_conv_1d_to_2d(WN.end)

    for i in range(len(WN.in_layers)):
        WN.in_layers[i] = convert_conv_1d_to_2d(WN.in_layers[i])

    for i in range(len(WN.res_skip_layers)):
        WN.res_skip_layers[i] = convert_conv_1d_to_2d(WN.res_skip_layers[i])

    # for i in range(len(WN.res_skip_layers)):
    WN.cond_layer = convert_conv_1d_to_2d(WN.cond_layer)


def convert_convinv_1d_to_2d(convinv):
    """
    Takes an invertible 1x1 1-d convolution and returns a 2-d convolution that does
    the inverse
    """
    conv2d = torch.nn.Conv2d(convinv.W_inverse.size(1),
                             convinv.W_inverse.size(0),
                             1, bias=False)
    conv2d.weight.data[:,:,:,0] = convinv.W_inverse.data
    return conv2d

def convert_1d_to_2d_(glow):
    """
    Caffe2 and TensorRT don't seem to support 1-d convolutions or properly
    convert ONNX exports with 1d convolutions to 2d convolutions yet, so we
    do the conversion to 2-d convolutions before ONNX export
    """
    # Convert upsample to 2d
    upsample = torch.nn.ConvTranspose2d(glow.upsample.weight.size(0),
                                        glow.upsample.weight.size(1),
                                        (glow.upsample.weight.size(2), 1),
                                        stride=(glow.upsample.stride[0], 1))
    upsample.weight.data[:, :, :, 0] = glow.upsample.weight.data
    upsample.bias.data = glow.upsample.bias.data
    glow.upsample = upsample.cuda()

    # Convert WN to 2d
    for WN in glow.WN:
        convert_WN_1d_to_2d_(WN)

    # Convert invertible conv to 2d
    for i in range(len(glow.convinv)):
        glow.convinv[i] = convert_convinv_1d_to_2d(glow.convinv[i])

    glow.cuda()


def infer_onnx(self, spect, z, sigma=0.9):
    spect = self.upsample(spect)
    # trim conv artifacts. maybe pad spec to kernel multiple
    time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
    spect = spect[:, :, :-time_cutoff]

    length_spect_group = spect.size(2) // 8
    mel_dim = 80
    batch_size = spect.size(0)

    spect = torch.squeeze(spect, 3)
    spect = spect.view((batch_size, mel_dim, length_spect_group, self.n_group))
    spect = spect.permute(0, 2, 1, 3)
    spect = spect.contiguous()
    spect = spect.view((batch_size, length_spect_group, self.n_group * mel_dim))
    spect = spect.permute(0, 2, 1)
    spect = torch.unsqueeze(spect, 3)
    spect = spect.contiguous()

    audio = z[:, :self.n_remaining_channels, :, :]
    z = z[:, self.n_remaining_channels:self.n_group, :, :]
    audio = sigma * audio

    for k in reversed(range(self.n_flows)):
        n_half = int(audio.size(1) / 2)
        audio_0 = audio[:, :n_half, :, :]
        audio_1 = audio[:, n_half:(n_half + n_half), :, :]

        output = self.WN[k]((audio_0, spect))
        s = output[:, n_half:(n_half + n_half), :, :]
        b = output[:, :n_half, :, :]
        audio_1 = (audio_1 - b) / torch.exp(s)
        audio = torch.cat([audio_0, audio_1], 1)

        audio = self.convinv[k](audio)

        if k % self.n_early_every == 0 and k > 0:
            audio = torch.cat((z[:, :self.n_early_size, :, :], audio), 1)
            z = z[:, self.n_early_size:self.n_group, :, :]

    audio = torch.squeeze(audio, 3)
    audio = audio.permute(0, 2, 1).contiguous().view(batch_size, (length_spect_group * self.n_group))

    return audio


@pytest.mark.usefixtures("neural_factory")
class TestDeployExport:
    @torch.no_grad()
    def __test_export_route(self, module, out_name, mode, input_example=None,
                            onnx_opset=None):
        # select correct extension based on the output format
        ext = {DF.ONNX: ".onnx", DF.TRTONNX: ".trt.onnx", DF.PYTORCH: ".pt", DF.TORCHSCRIPT: ".ts"}.get(mode, ".onnx")
        out = Path(f"{out_name}{ext}")
        out_name = str(out)

        if out.exists():
            os.remove(out)

        # module.eval()
        module.waveglow.train(False)

        # for name, mdl in module.waveglow.named_children():
        #     print(name, " <=====> ", mdl)

        # torch.manual_seed(10)
        if isinstance(input_example, OrderedDict):
            outputs_fwd = module.forward(*tuple(input_example.values()))
        elif isinstance(input_example, tuple): #or isinstance(input_example, list)
            outputs_fwd = module.forward(*input_example)
        elif input_example is not None:
            outputs_fwd = module.waveglow.infer(input_example)
        else:
            outputs_fwd = None

        deploy_input_example = (
            tuple(input_example.values()) if isinstance(input_example, OrderedDict) else input_example
        )
        deploy_output_example = (
            outputs_fwd[0] if isinstance(outputs_fwd, tuple) else outputs_fwd
        )
        self.nf.deployment_export(
            module=module,
            output=out_name,
            input_example=deploy_input_example,
            d_format=mode,
            # output_example=outputs_fwd, #deploy_output_example,
            output_example=deploy_output_example,
            onnx_opset = onnx_opset
        )

        tol = 5.0e-3
        assert out.exists() == True

        if mode == DF.TRTONNX:

            data_loader = DefaultDataLoader(10)
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
                    inputs[input_name] = (
                        input_example[input_name].cpu().numpy()
                        if isinstance(input_example, OrderedDict)
                        else (
                            input_example[i].cpu().numpy()
                            if isinstance(input_example, tuple)
                            else input_example.cpu().numpy()
                        )
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
            torch.manual_seed(10)
            outputs_fwd = module.forward(*inpex)

        elif mode == DF.ONNX:
            # module.eval()
            # Must recompute because *module* might be different now
            # torch.manual_seed(10)
            if isinstance(input_example, OrderedDict):
                outputs_fwd = module.forward(*tuple(input_example.values()))
            elif isinstance(input_example, tuple):  # or isinstance(input_example, list)
                outputs_fwd = module.forward(*input_example)
            elif input_example is not None:
                outputs_fwd = module.forward(input_example)
            else:
                outputs_fwd = None

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            ort_session = ort.InferenceSession(out_name, sess_options, ['CUDAExecutionProvider'])
            print('Execution Providers: ', ort_session.get_providers())
            inputs = dict()
            input_names = list(module.input_ports)
            ort_inputs = ort_session.get_inputs()
            for i in range(len(input_names)):
                input_name = input_names[i]
                if input_name in module._disabled_deployment_input_ports:
                    continue
                ort_name = input_name
                for node_arg in ort_inputs:
                    if node_arg.name[:len(input_name)] == input_name:
                        ort_name = node_arg.name
                        break

                inputs[ort_name] = (
                    input_example[input_name].cpu().numpy()
                    if isinstance(input_example, OrderedDict)
                    else (
                        input_example[i].cpu().numpy()
                        if isinstance(input_example, tuple)
                        else input_example.cpu().numpy()
                    )
                )

            output_names = list(module.output_ports.keys())
            output_names = output_names[0]

            # try:
            outputs_scr = ort_session.run(output_names, inputs)

            # except ort.capi.onnxruntime_pybind11_state.InvalidArgument as e:
            #     for node_arg in ort_inputs:
            #         print(node_arg.name)

            outputs_scr = torch.from_numpy(outputs_scr[0]).cuda()
        elif mode == DF.TORCHSCRIPT:
            tscr = torch.jit.load(out_name)
            # if isinstance(module, nemo.backends.pytorch.tutorials.TaylorNet):
            #     input_example = torch.randn(4, 1).cuda()
            # outputs_fwd = module.forward(input_example)
            torch.manual_seed(10)
            outputs_scr = (
                tscr.forward(*tuple(input_example.values()))
                if isinstance(input_example, OrderedDict)
                else (
                    tscr.forward(*input_example)
                    if isinstance(input_example, tuple)
                    else tscr.forward(input_example)
                )
            )
        elif mode == DF.PYTORCH:
            module.load_state_dict(torch.load(out_name))
            # module.eval()
            torch.manual_seed(10)
            if isinstance(input_example, OrderedDict):
                outputs_scr = module.forward(*tuple(input_example.values()))
            elif isinstance(input_example, tuple) or isinstance(input_example, list):
                outputs_scr = module.forward(*input_example)
            else:
                outputs_scr = module.forward(input_example)

            #
            #     outputs_scr = (
            #     module.forward(*tuple(input_example.values()))
            #     if isinstance(input_example, OrderedDict)
            #     else (
            #         module.forward(*input_example)
            #         if isinstance(input_example, tuple) or isinstance(input_example, list)
            #         else module.forward(input_example)
            #     )
            # )

        outputs_scr = (
            outputs_scr[0] if isinstance(outputs_scr, tuple) or isinstance(outputs_scr, list) else outputs_scr
        )
        outputs_fwd = (
            outputs_fwd[0] if isinstance(outputs_fwd, tuple) or isinstance(outputs_fwd, list) else outputs_fwd
        )

        # print("outputs_scr\n", str(outputs_scr)[:200])
        # print("outputs_fwd\n", str(outputs_fwd)[:200])

        assert (outputs_scr - outputs_fwd).norm(p=2) < tol

        if out.exists():
            os.remove(out)

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.parametrize(
        "input_example, module_name, df_type, onnx_opset",
        [
            # # TaylorNet export tests.
            # (torch.randn(4, 1).cuda(), "TaylorNet", DF.PYTORCH, None),
            # # TokenClassifier export tests.
            # (torch.randn(16, 16, 512).cuda(), "TokenClassifier", DF.ONNX, 9),
            # (torch.randn(16, 16, 512).cuda(), "TokenClassifier", DF.TORCHSCRIPT, None),
            # (torch.randn(16, 16, 512).cuda(), "TokenClassifier", DF.PYTORCH, None),
            # pytest.param(torch.randn(16, 16, 512).cuda(), "TokenClassifier", DF.TRTONNX, 9, marks=requires_trt),
            # # JasperDecoderForCTC export tests.
            # (torch.randn(34, 1024, 1).cuda(), "JasperDecoderForCTC", DF.ONNX, 11),
            # (torch.randn(34, 1024, 1).cuda(), "JasperDecoderForCTC", DF.TORCHSCRIPT, None),
            # (torch.randn(34, 1024, 1).cuda(), "JasperDecoderForCTC", DF.PYTORCH, None),
            # pytest.param(torch.randn(34, 1024, 1).cuda(), "JasperDecoderForCTC", DF.TRTONNX, 11, marks=requires_trt),
            # # JasperEncoder export tests.
            # (torch.randn(16, 64, 256).cuda(), "JasperEncoder", DF.ONNX, 11),
            # (torch.randn(16, 64, 256).cuda(), "JasperEncoder", DF.TORCHSCRIPT, None),
            # (torch.randn(16, 64, 256).cuda(), "JasperEncoder", DF.PYTORCH, None),
            # pytest.param(torch.randn(16, 64, 256).cuda(), "JasperEncoder", DF.TRTONNX, 11, marks=requires_trt),
            # # QuartznetEncoder export tests.
            # (torch.randn(16, 64, 256).cuda(), "QuartznetEncoder", DF.ONNX, 11),
            # (torch.randn(16, 64, 256).cuda(), "QuartznetEncoder", DF.TORCHSCRIPT, None),
            # (torch.randn(16, 64, 256).cuda(), "QuartznetEncoder", DF.PYTORCH, None),
            # pytest.param(torch.randn(16, 64, 256).cuda(), "QuartznetEncoder", DF.TRTONNX, 11, marks=requires_trt),
        ],
    )
    def test_module_export(self, tmpdir, input_example, module_name, df_type, onnx_opset):
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
        # Test export.
        self.__test_export_route(
            module=module, out_name=tmp_file_name, mode=df_type, input_example=input_example, onnx_opset=onnx_opset
        )

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.parametrize(
        "df_type", [DF.ONNX, DF.TORCHSCRIPT, DF.PYTORCH, pytest.param(DF.TRTONNX, marks=requires_trt)]
    )
    def OOtest_hf_bert(self, tmpdir, df_type):
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
        self.__test_export_route(module=bert, out_name=tmp_file_name, mode=df_type,
                                 input_example=input_example, onnx_opset=11)


    # @torch.no_grad()
    # def test_waveglow(self):
    #     vocoder_model_config = "examples/tts/configs/waveglow.yaml"
    #     waveglow_sigma = 0.6
    #
    #     waveglow = nemo_tts.WaveGlowInferNM.import_from_config(
    #         vocoder_model_config, "WaveGlowInferNM",
    #         overwrite_params={
    #             "sigma": waveglow_sigma,
    #             "sample_rate": 16000,
    #         }
    #     )
    #
    #     torch.manual_seed(10)
    #     mel_spectrogram=torch.rand(size=(4, 80, 10), dtype=torch.float).cuda()
    #     # audio_inp=torch.rand(size=(4, 2560), dtype=torch.float).cuda()
    #
    #     #     [
    #     #         ("mel_spectrogram", mel_spectrogram),
    #     #         # ("audio", audio_inp),
    #     #     ]
    #     # )
    #
    #     self.__test_export_route(module=waveglow, #.inner_glow(),
    #                              out_name="waveglow", mode=DF.ONNX, #TORCHSCRIPT,
    #                              input_example = mel_spectrogram, onnx_opset=11) #[mel_spectrogram.cuda(), audio_inp.cuda()]) #



    @torch.no_grad()
    def test_waveglow(self):
        # curl - LO
        url = "https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ljs_256channels/versions/2/files/waveglow_256channels_ljs_v2.pt"
        ptfile = "./waveglow_256channels_ljs_v2.pt"
        if not Path(ptfile).is_file():
            urllib.request.urlretrieve(url, ptfile)

        waveglow = load_and_setup_model(ptfile,
            forward_is_infer=False)

        output = "./waveglow.onnx"
        # 80 mel channels, 620 mel spectrograms ~ 7 seconds of speech
        mel = torch.randn(1, 80, 620).cuda()
        stride = 256 # value from waveglow upsample
        n_group = 8
        z_size2 = (mel.size(2)*stride)//n_group
        z = torch.randn(1, n_group, z_size2, 1).cuda()
        sigma_infer = 1.0

        with torch.no_grad():
            # run inference to force calculation of inverses
            waveglow.infer(mel, sigma=sigma_infer)

            # export to ONNX
            convert_1d_to_2d_(waveglow)
            fType = types.MethodType
            waveglow.forward = fType(infer_onnx, waveglow)
            mel = mel.unsqueeze(3)
            opset_version = 10 #11

            torch.onnx.export(waveglow, (mel, z), output,
                              opset_version=opset_version,
                              do_constant_folding=True,
                              input_names=["mel", "z"],
                              output_names=["audio"],
                              dynamic_axes={"mel":   {0: "batch_size", 2: "mel_seq"},
                                            "z":     {0: "batch_size", 2: "z_seq"},
                                            "audio": {0: "batch_size", 1: "audio_seq"}})

