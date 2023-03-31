# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import numpy as np
import pytest

from nemo.collections.tts.parts.preprocessing.features import MelSpectrogramFeaturizer, PitchFeaturizer, compute_energy


class TestTTSFeatures:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_compute_energy(self):
        input_spec = [[[0.0, 0.0, 0.0], [1.0, 0.5, 2.0]], [[3.0, 0.0, 1.0], [0.5, 0.5, 2.0]]]
        input_spec = np.array(input_spec).transpose((0, 2, 1))
        expected_energy = [[0.0, np.sqrt(5.25)], [np.sqrt(10.0), np.sqrt(4.5)]]
        expected_energy = np.array(expected_energy)

        energy = compute_energy(input_spec)
        np.testing.assert_array_almost_equal(energy, expected_energy)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_compute_mel_spectrogram(self):
        spec_dim = 40
        audio_len = 10000
        hop_len = 100
        expected_spec_len = 1 + (audio_len // hop_len)
        input_audio = np.random.uniform(size=[audio_len])

        mel_featurizer = MelSpectrogramFeaturizer(mel_dim=spec_dim, hop_length=hop_len)
        spec = mel_featurizer.compute_mel_spectrogram(input_audio)

        assert len(spec.shape) == 2
        assert spec.shape[0] == spec_dim
        assert spec.shape[1] == expected_spec_len

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_compute_pitch(self):
        audio_len = 10000
        hop_len = 100
        expected_spec_len = 1 + (audio_len // hop_len)
        input_audio = np.random.uniform(size=[audio_len])

        pitch_featurizer = PitchFeaturizer(hop_length=hop_len)
        pitch, voiced, voiced_prob = pitch_featurizer.compute_pitch(input_audio)

        assert len(pitch.shape) == 1
        assert pitch.shape[0] == expected_spec_len
        assert pitch.dtype == float

        assert len(voiced.shape) == 1
        assert voiced.shape[0] == expected_spec_len
        assert voiced.dtype == bool

        assert len(voiced_prob.shape) == 1
        assert voiced_prob.shape[0] == expected_spec_len
        assert voiced_prob.dtype == float
