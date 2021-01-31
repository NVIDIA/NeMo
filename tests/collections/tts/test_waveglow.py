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
from unittest import TestCase

import onnx
import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.tts.models import WaveGlowModel
from nemo.collections.tts.modules import WaveGlowModule

try:
    import tensorrt as trt

    print("TRT is available!\n")
    trt_available = True
except Exception as e:
    print("TRT is not available!\n")
    trt_available = False

mcfg = DictConfig(
    {
        "_target_": "nemo.collections.tts.modules.waveglow.WaveGlowModule",
        "n_flows": 12,
        "n_group": 8,
        "n_mel_channels": 80,
        "n_early_every": 4,
        "n_early_size": 2,
        "n_wn_channels": 512,
        "n_wn_layers": 8,
        "wn_kernel_size": 3,
    }
)

pcfg = DictConfig(
    {
        "_target_": "nemo.collections.asr.parts.features.FilterbankFeatures",
        "dither": 0.0,
        "nfilt": 80,
        "stft_conv": True,
    }
)

wcfg = DictConfig({"waveglow": mcfg, "sigma": 1.0, "preprocessor": pcfg,})


def input_example(sz):
    mel = torch.randn(1, 80, sz).cuda().half()
    z = torch.randn(1, 8, sz * 256 // 8).cuda().half()
    return {"spec": mel, "z": z}


class TestWaveGlow:
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_export_to_onnx(self):
        model = WaveGlowModel(wcfg)
        # model = WaveGlowModel.restore_from("../WaveGlow-22050Hz-268M.nemo")
        model = model.cuda().half()
        with tempfile.TemporaryDirectory() as tmpdir, model.nemo_infer():
            # Generate filename in the temporary directory.
            tmp_file_name = os.path.join("waveglow.onnx")

            n_mels = 80
            # Test export.
            inp = input_example(n_mels)

            inp2 = inp
            inp3 = inp2
            res1 = model.waveglow(**inp)
            res2 = model.waveglow(**inp2)
            assert torch.allclose(res1, res2, rtol=0.01, atol=0.1)

            model.export(
                tmp_file_name,
                verbose=True,
                input_example=inp3,
                output_example=res1,
                try_script=False,
                check_trace=False,
                do_constant_folding=True,
                use_dynamic_axes=False,
            )

            try:
                test_runtime = True
                import onnxruntime
            except (ImportError, ModuleNotFoundError):
                test_runtime = False
            if test_runtime:
                omodel = onnx.load(tmp_file_name)
                output_names = ['audio']
                sess = onnxruntime.InferenceSession(omodel.SerializeToString())
                output = sess.run(None, {"spec": inp["spec"].cpu().numpy(), "z": inp["z"].cpu().numpy()})[0]
                assert torch.allclose(torch.from_numpy(output), res2.cpu(), rtol=1, atol=100)

            if False:  # trt_available:
                opt_profiles = [
                    [
                        {"input": "spec", "min": (1, 80, n_mels), "opt": (1, 80, n_mels), "max": (1, 80, n_mels)},
                        {
                            "input": "z",
                            "min": (1, 8, n_mels * 256 // 8),
                            "opt": (1, 8, n_mels * 256 // 8),
                            "max": (1, 8, n_mels * 256 // 8),
                        },
                    ]
                ]
                engine = build_trt_engine_from_onnx(tmp_file_name, opt_profiles, verbose=True)
                assert engine is not None
                with open(tmp_file_name + '.trt', 'wb') as f:
                    f.write(engine.serialize())


def build_trt_engine_from_onnx(onnx_path, opt_profiles, verbose=False):
    """Builds TRT engine from an ONNX file
		"""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    builder.max_batch_size = 1

    with open(onnx_path, 'rb') as model_fh:
        model = model_fh.read()
        model_onnx = onnx.load_model_from_string(model)

    if True:  # self.model.args.fp16:
        builder.fp16_mode = True
        config_flags = 1 << int(trt.BuilderFlag.FP16)  # | 1 << int(trt.BuilderFlag.STRICT_TYPES)
    else:
        config_flags = 0

    builder.max_workspace_size = 256 * 1024 * 1024

    config = builder.create_builder_config()
    config.flags = config_flags
    config.max_workspace_size = 1 << 29

    for x in opt_profiles:
        profile = builder.create_optimization_profile()
        for p in x:
            profile.set_shape(**p)
        config.add_optimization_profile(profile)

    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)

    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        parsed = parser.parse(model)
        print(f"Parsing returned {parsed}, building TRT engine ...")
        return builder.build_engine(network, config=config)


if __name__ == "__main__":
    t = TestWaveGlow()
    t.test_export_to_onnx()
