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

import pytest
import torch

from nemo.collections.tts.torch.data import CharMelAudioDataset


class TestCharDataset:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    @pytest.mark.torch_tts
    def test_dataset(self, test_data_dir):
        manifest_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/manifest.json')
        sup_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/sup')

        dataset = CharMelAudioDataset(
            manifest_filepath=manifest_path, sample_rate=22050, supplementary_folder=sup_path
        )

        dataloader = torch.utils.data.DataLoader(dataset, 2, collate_fn=dataset._collate_fn)

        data, _, _, _, _, _, _ = next(iter(dataloader))


class TestPhoneDataset:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    @pytest.mark.torch_tts
    def test_dataset(self, test_data_dir):
        manifest_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/manifest.json')
        sup_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/sup')

        dataset = CharMelAudioDataset(
            manifest_filepath=manifest_path, sample_rate=22050, supplementary_folder=sup_path
        )

        dataloader = torch.utils.data.DataLoader(dataset, 2, collate_fn=dataset._collate_fn)

        _, _, _, _, _, _, _ = next(iter(dataloader))
