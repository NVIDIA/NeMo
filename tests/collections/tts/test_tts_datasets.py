# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.tts.data.datalayers import FastSpeech2Dataset


class TestTTSDatasets:
    @pytest.mark.unit
    def test_fs2_dataset(self, test_data_dir):
        manifest_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/manifest.json')
        mappings_file = os.path.join(test_data_dir, 'tts/mini_ljspeech/mappings.json')
        ignore_file = os.path.join(test_data_dir, 'tts/mini_ljspeech/wavs_to_ignore.pkl')

        # Test loading data (including supplementary data) with ignore file
        ds = FastSpeech2Dataset(
            manifest_filepath=manifest_path,
            mappings_filepath=mappings_file,
            sample_rate=22050,
            ignore_file=ignore_file,
            load_supplementary_values=True,
        )

        assert len(ds) == 4
        count = 0
        for _ in ds:
            count += 1
        assert count == 4
