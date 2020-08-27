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

from nemo.collections.tts.models import WaveGlowModel
from nemo.collections.tts.modules import WaveGlowModule

wcfg = DictConfig(
    {
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


class TestWaveGlow:
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_export_to_onnx(self):
        model = WaveGlowModule(**wcfg).cuda().half()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate filename in the temporary directory.
            tmp_file_name = os.path.join("waveglow.onnx")
            # Test export.
            model.export(tmp_file_name, check_trace=False)
