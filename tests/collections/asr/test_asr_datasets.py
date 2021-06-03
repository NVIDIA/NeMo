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

import copy
import os

import pytest
from omegaconf import OmegaConf

from nemo.collections.asr.data.audio_to_text import TarredAudioToBPEDataset, TarredAudioToCharDataset
from nemo.collections.asr.data.audio_to_text_dataset import inject_dataloader_value_from_model_config
from nemo.collections.common import tokenizers
from nemo.utils import logging


class TestASRDatasets:
    labels = [
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "'",
    ]

    @pytest.mark.unit
    def test_tarred_dataset(self, test_data_dir):
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/tarred_an4/tarred_audio_manifest.json'))

        # Test braceexpand loading
        tarpath = os.path.abspath(os.path.join(test_data_dir, 'asr/tarred_an4/audio_{0..1}.tar'))
        ds_braceexpand = TarredAudioToCharDataset(
            audio_tar_filepaths=tarpath, manifest_filepath=manifest_path, labels=self.labels, sample_rate=16000
        )
        assert len(ds_braceexpand) == 32
        count = 0
        for _ in ds_braceexpand:
            count += 1
        assert count == 32

        # Test loading via list
        tarpath = [os.path.abspath(os.path.join(test_data_dir, f'asr/tarred_an4/audio_{i}.tar')) for i in range(2)]
        ds_list_load = TarredAudioToCharDataset(
            audio_tar_filepaths=tarpath, manifest_filepath=manifest_path, labels=self.labels, sample_rate=16000
        )
        count = 0
        for _ in ds_list_load:
            count += 1
        assert count == 32

    @pytest.mark.unit
    def test_mismatch_in_model_dataloader_config(self, caplog):
        logging._logger.propagate = True
        caplog.set_level(logging.WARNING)

        model_cfg = OmegaConf.create(dict(labels=OmegaConf.create(["a", "b", "c"])))
        dataloader_cfg = OmegaConf.create(dict(labels=copy.deepcopy(self.labels)))

        inject_dataloader_value_from_model_config(model_cfg, dataloader_cfg, key='labels')

        assert (
            """`labels` is explicitly provided to the data loader, and is different from the `labels` provided at the model level config."""
            in caplog.text
        )

        logging._logger.propagate = False

    @pytest.mark.unit
    def test_tarred_bpe_dataset(self, test_data_dir):
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/tarred_an4/tarred_audio_manifest.json'))

        tokenizer_path = os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128", 'vocab.txt')
        tokenizer = tokenizers.AutoTokenizer(pretrained_model_name='bert-base-cased', vocab_file=tokenizer_path)

        # Test braceexpand loading
        tarpath = os.path.abspath(os.path.join(test_data_dir, 'asr/tarred_an4/audio_{0..1}.tar'))
        ds_braceexpand = TarredAudioToBPEDataset(
            audio_tar_filepaths=tarpath, manifest_filepath=manifest_path, tokenizer=tokenizer, sample_rate=16000
        )
        assert len(ds_braceexpand) == 32
        count = 0
        for _ in ds_braceexpand:
            count += 1
        assert count == 32

        # Test loading via list
        tarpath = [os.path.abspath(os.path.join(test_data_dir, f'asr/tarred_an4/audio_{i}.tar')) for i in range(2)]
        ds_list_load = TarredAudioToBPEDataset(
            audio_tar_filepaths=tarpath, manifest_filepath=manifest_path, tokenizer=tokenizer, sample_rate=16000
        )
        count = 0
        for _ in ds_list_load:
            count += 1
        assert count == 32
