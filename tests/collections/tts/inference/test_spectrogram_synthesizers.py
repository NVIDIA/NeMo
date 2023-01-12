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

from nemo.collections.tts.inference.spectrogram_synthesizers import FastPitchSpectrogramSynthesizer
from nemo.collections.tts.models import FastPitchModel


class TestSpectrogramSynthesizers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestSpectrogramSynthesizers, cls).setUpClass()

        fastpitch_model = FastPitchModel.from_pretrained("tts_en_fastpitch_multispeaker").eval().to("cpu")
        cls.spec_synthesizer = FastPitchSpectrogramSynthesizer(model=fastpitch_model)

    @pytest.mark.with_downloads()
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_synthesize_spectrogram(self):
        batch_size = 2
        text_length = 10
        spec_dim = 80
        tokens = torch.zeros([batch_size, text_length], dtype=torch.int32).to("cpu")
        speaker = torch.randint(low=0, high=10, size=[batch_size], dtype=torch.int32).to("cpu")
        pitch = torch.rand([batch_size, text_length]).to("cpu")
        spec = self.spec_synthesizer.synthesize_spectrogram(tokens=tokens, speaker=speaker, pitch=pitch)

        assert len(spec.shape) == 3
        assert spec.shape[0] == batch_size
        assert spec.shape[1] == spec_dim
        assert spec.shape[2] > text_length
