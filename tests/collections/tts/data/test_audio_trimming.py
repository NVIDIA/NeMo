# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.collections.tts.data.audio_trimming import get_start_and_end_of_speech, pad_sample_indices


class TestAudioTrimming:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_start_and_end_of_speech(self):
        # First speech frame is index 2 (samples 200-300) and last one is index 7 (samples 700-800).
        is_speech = np.array([True, False, True, True, False, True, True, True, False, True, False])
        frame_threshold = 2
        frame_step = 100

        start_i, end_i = get_start_and_end_of_speech(
            is_speech=is_speech, frame_threshold=frame_threshold, frame_step=frame_step
        )

        assert start_i == 200
        assert end_i == 800

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_start_and_end_of_speech_not_found(self):
        is_speech = np.array([False, True, True, False])
        frame_threshold = 3
        frame_step = 100

        start_i, end_i = get_start_and_end_of_speech(
            is_speech=is_speech, frame_threshold=frame_threshold, frame_step=frame_step, audio_id="test"
        )

        assert start_i == 0
        assert end_i == 400

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_pad_sample_indices(self):
        start_i, end_i = pad_sample_indices(
            start_sample_i=1000, end_sample_i=2000, max_sample=5000, sample_rate=100, pad_seconds=3
        )
        assert start_i == 700
        assert end_i == 2300

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_pad_sample_indices_boundaries(self):
        start_i, end_i = pad_sample_indices(
            start_sample_i=100, end_sample_i=1000, max_sample=1150, sample_rate=100, pad_seconds=2
        )
        assert start_i == 0
        assert end_i == 1150
