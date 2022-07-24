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
import io
import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Union

import braceexpand
import numpy as np
import torch
import webdataset as wd
from torch.utils.data import ChainDataset

from nemo.collections.asr.data.audio_to_text import AudioToCharDataset
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.common import tokenizers
from nemo.collections.common.parts.preprocessing import collections, parsers
from nemo.core.classes import Dataset, IterableDataset
from nemo.core.neural_types import *
from nemo.utils import logging

__all__ = [
    'AudioToAudioDataset',
]


class _AudioDataset(Dataset):
    """
    """

    def __init__(
        self,
        manifest_filepath: str,
        num_sources: int,
        featurizer,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_utts: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        if type(manifest_filepath) == str:
            manifest_filepath = manifest_filepath.split(',')

        self.collection = collections.SpeechSeparationAudio(
            manifest_files=manifest_filepath,
            num_sources=num_sources,
            min_duration=min_duration,
            max_duration=max_duration,
            max_utts=max_utts,
        )

        self.num_sources = num_sources
        self.featurizer = featurizer
        self.orig_sr = kwargs.get('orig_sr', None)

    def get_manifest_sample(self, sample_id):
        return self.collection[sample_id]

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        sample = self.collection[index]

        features_list = [
            self.featurizer.process(
                sample.audio_file[i], offset=sample.offset[i], duration=sample.duration[i], orig_sr=self.orig_sr,
            )
            * sample.scale_factor[i]
            for i in range(self.num_sources)
        ]

        return {
            'features_list': features_list,
        }


class AudioToSourceDataset(_AudioDataset):
    """
    AudioToAudioDataset is intended for Audio to Audio tasks such as speech separation,
    speech enchancement, music source separation etc.
    """

    def __init__(
        self,
        manifest_filepath: str,
        num_sources: int,
        featurizer,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            manifest_filepath=manifest_filepath,
            num_sources=num_sources,
            featurizer=featurizer,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            **kwargs,
        )

        self.num_sources = num_sources

    def __getitem__(self, index):
        data_pt = super().__getitem__(index)
        features_list = data_pt['features_list']
        features_lengths = [torch.tensor(x.shape[0]).long() for x in features_list]

        min_l = torch.min(torch.stack(features_lengths)).item()

        if self.num_sources == 2:
            t1, t2 = [x[:min_l] for x in features_list]
            mix = t1 + t2
            output = [mix, torch.tensor(min_l).long(), t1, t2]
        elif self.num_sources == 3:
            t1, t2, t3 = [x[:min_l] for x in features_list]
            mix = t1 + t2 + t3
            output = [mix, torch.tensor(min_l).long(), t1, t2, t3]

        # add index
        output.append(index)

        return output

    def _collate_fn(self, batch):
        return _audio_to_audio_collate_fn(batch)


def _audio_to_audio_collate_fn(batch):
    """collate batch 
    Args:
        batch 
        This collate func assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    packed_batch = list(zip(*batch))
    if len(packed_batch) == 5:
        _, audio_lengths, _, _, sample_ids = packed_batch
    elif len(packed_batch) == 6:
        _, audio_lengths, _, _, _, sample_ids = packed_batch
    else:
        raise ValueError("Expects 5 or 6 tensors in the batch!")

    # convert sample_ids to torch
    sample_ids = torch.tensor(sample_ids, dtype=torch.int32)

    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()

    audio_signal, target1, target2, target3 = [], [], [], []
    for b in batch:
        if len(b) == 5:
            sig, sig_len, t1, t2, _ = b
        else:
            sig, sig_len, t1, t2, t3, _ = b
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
                t1 = torch.nn.functional.pad(t1, pad)
                t2 = torch.nn.functional.pad(t2, pad)
                if len(b) == 6:
                    t3 = torch.nn.functional.pad(t3, pad)

            audio_signal.append(sig)
            target1.append(t1)
            target2.append(t2)
            if len(b) == 6:
                target3.append(t3)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        target1 = torch.stack(target1)
        target2 = torch.stack(target2)
        if len(b) == 6:
            target3 = torch.stack(target3)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, target1, target2, target3, audio_lengths, sample_ids = None, None, None, None, None, None

    if len(packed_batch) == 5:
        return audio_signal, audio_lengths, target1, target2, sample_ids
    else:
        return audio_signal, audio_lengths, target1, target2, target3, sample_ids


def test_AudioToSourceDataset():
    featurizer = WaveformFeaturizer(sample_rate=16000)
    dataset = AudioToSourceDataset(
        manifest_filepath='/media/kpuvvada/data/datasets/data/wsj0-2mix/manifests/test.json',
        num_sources=2,
        featurizer=featurizer,
    )

    item = dataset.__getitem__(1)
    print([x for x in item])


if __name__ == '__main__':
    import nemo

    test_AudioToSourceDataset()
