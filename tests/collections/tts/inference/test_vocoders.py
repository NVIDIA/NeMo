# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import pytest
import unittest

import torch

from nemo.collections.tts.inference.vocoders import HifiGanVocoder
from nemo.collections.tts.models import HifiGanModel


class TestVocoders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestVocoders, cls).setUpClass()

        hifigan_model = HifiGanModel.from_pretrained("tts_en_hifitts_hifigan_ft_fastpitch").eval().to("cpu")
        cls.vocoder = HifiGanVocoder(model=hifigan_model)

    @pytest.mark.with_downloads()
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_convert_spectrogram_to_audio(self):
        batch_size = 2
        spec_dim = 80
        spec_length = 100
        spec = torch.rand(size=[batch_size, spec_dim, spec_length], dtype=torch.float32).to("cpu")
        audio = self.vocoder.convert_spectrogram_to_audio(spec=spec)

        assert len(audio.shape) == 2
        assert audio.shape[0] == batch_size
        assert audio.shape[1] > spec_length
