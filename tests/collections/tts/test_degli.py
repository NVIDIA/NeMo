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

import pytest
from omegaconf import DictConfig

from nemo.collections.tts.models import DegliModel
from nemo.collections.tts.modules import DegliModule

wcfg = DictConfig(
    {
        "n_fft": 512,
        "hop_length": 256,
        "depth": 1,
        "out_all_block": True,
        "tiny": False,
        "layers": 6,
        "k_x": 3,
        "k_y": 3,
        "s_x": 1,
        "s_y": 2,
        "widening": 16,
        "use_bn": False,
        "linear_finalizer": True,
        "convGlu": False,
        "act": "relu",
        "act2": "selu",
        "glu_bn": True,
        "use_weight_norm": True,
    }
)


class TestDegli:
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_export_to_onnx(self):
        model = DegliModule(**wcfg).cuda()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate filename in the temporary directory.
            tmp_file_name = os.path.join("degli.onnx")
            # Test export.
            model.export(tmp_file_name, check_trace=False)
