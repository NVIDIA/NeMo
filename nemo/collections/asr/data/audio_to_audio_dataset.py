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

import json
import random
from typing import Any, List, Optional, Union

import torch
from omegaconf import DictConfig, open_dict
from omegaconf.listconfig import ListConfig
from pytorch_lightning.callbacks import BasePredictionWriter
from torch.utils.data import ChainDataset

from nemo.collections.asr.data import audio_to_audio
from nemo.utils import logging


def get_audio_to_source_dataset(config: dict, featurizer,) -> audio_to_audio.AudioToSourceDataset:
    """
    Instantiates an audio to source(s) dataset.
    Args:
        config: Config of the AudioToSourceDataset.
        featurizer: An instance of featurizer.
    Returns:
        An instance of AudioToSourceDataset.
    """
    dataset = audio_to_audio.AudioToSourceDataset(
        manifest_filepath=config['manifest_filepath'],
        num_sources=config['num_sources'],
        featurizer=featurizer,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        trim=config.get('trim_silence', False),
        orig_sr=config.get('orig_sr', None),
    )
    return dataset
