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

import pytest
import unittest
import torch
import numpy as np

from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.tts.inference.audio_processors import MelSpectrogramProcessor


class TestAudioProcessors(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestAudioProcessors, cls).setUpClass()

        cls.spec_dim = 80
        audio_mel_processor = AudioToMelSpectrogramPreprocessor(
            sample_rate=44100,
            features=cls.spec_dim,
            n_window_size=2048,
            n_window_stride=512,
            window_size=False,
            window_stride=False,
            n_fft=2048,
            lowfreq=0,
            highfreq=None,
            pad_to=0
        )
        cls.audio_processor = MelSpectrogramProcessor(preprocessor=audio_mel_processor)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_compute_spectrogram(self):
        batch_size = 2
        audio_len1 = 10000
        audio_len2 = 20000
        audio_tensor = torch.rand(size=[batch_size, audio_len2], dtype=torch.float32).to("cpu")
        audio_len_tensor = torch.tensor([audio_len1, audio_len2]).to("cpu")

        spec, spec_len = self.audio_processor.compute_spectrogram(audio=audio_tensor, audio_len=audio_len_tensor)

        assert len(spec.shape) == 3
        assert len(spec_len.shape) == 1
        assert spec.shape[0] == batch_size
        assert spec.shape[1] == self.spec_dim
        assert spec.shape[2] == spec_len[1]
        # Validate padded outputs are 0
        np.testing.assert_array_almost_equal(spec[0][:, spec_len[0]:], 0.0)

