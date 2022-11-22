# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from abc import ABC
from itertools import chain, cycle
from typing import List

import numpy as np
from torch.utils.data import IterableDataset

from nemo.collections.asr.data.deep_diarize.train_data import LocalRTTMStreamingSegmentsDataset
from nemo.collections.asr.data.deep_diarize.utils import ContextWindow
from nemo.collections.asr.data.deep_diarize.vox_celeb_train_data import VoxCelebConfig, VoxCelebDataset
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.modules.audio_preprocessing import SpectrogramAugmentation
from nemo.collections.asr.parts.preprocessing import WaveformFeaturizer


class ConcatDataset(IterableDataset):
    def __init__(self, datasets, batch_size, weights):
        self.datasets = [iter(dataset) for dataset in datasets]
        self.batch_size = batch_size
        self.weights = weights

    def process_data(self, x):
        dataset = np.random.choice(self.datasets, 1, p=self.weights)[0]
        train_segment, train_length, targets, start_segment = next(dataset)
        yield train_segment, train_length, targets, start_segment
        start_segment = False
        while not start_segment:
            train_segment, train_length, targets, start_segment = next(dataset)
            yield train_segment, train_length, targets, start_segment

    def get_stream(self):
        return chain.from_iterable(map(self.process_data, cycle([None])))

    def get_streams(self):
        return zip(*[self.get_stream() for _ in range(self.batch_size)])

    def __iter__(self):
        return self.get_streams()


class CombinedSegmentDataset(IterableDataset, ABC):
    @classmethod
    def create_streaming_datasets(
        cls,
        batch_size,
        manifest_filepaths: List[str],
        preprocessor: AudioToMelSpectrogramPreprocessor,
        featurizer: WaveformFeaturizer,
        context_window: ContextWindow,
        spec_augmentation: SpectrogramAugmentation,
        voxceleb_config: VoxCelebConfig,
        window_stride: float,
        subsampling: int,
        train_segment_seconds: int,
        max_workers,
        weights: List[float],
        max_speakers: int,
    ):
        num_workers = max_workers
        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers = n
                break
        split_size = batch_size // num_workers
        return cls.create_datasets(
            featurizer=featurizer,
            manifest_filepaths=manifest_filepaths,
            voxceleb_config=voxceleb_config,
            num_workers=num_workers,
            preprocessor=preprocessor,
            context_window=context_window,
            spec_augmentation=spec_augmentation,
            split_size=split_size,
            subsampling=subsampling,
            train_segment_seconds=train_segment_seconds,
            window_stride=window_stride,
            weights=weights,
            max_speakers=max_speakers,
        )

    @classmethod
    def create_datasets(
        cls,
        featurizer: WaveformFeaturizer,
        manifest_filepaths: List[str],
        num_workers: int,
        preprocessor: AudioToMelSpectrogramPreprocessor,
        spec_augmentation: SpectrogramAugmentation,
        voxceleb_config: VoxCelebConfig,
        context_window: ContextWindow,
        split_size: int,
        subsampling: int,
        train_segment_seconds: int,
        window_stride: float,
        weights: List[float],
        max_speakers: int,
    ):
        datasets = []
        for manifest_filepath in manifest_filepaths:
            calls = LocalRTTMStreamingSegmentsDataset.data_setup(
                manifest_filepath=manifest_filepath, max_speakers=max_speakers
            )
            datasets.append(
                LocalRTTMStreamingSegmentsDataset(
                    data_list=calls,
                    manifest_filepath=manifest_filepath,
                    preprocessor=preprocessor,
                    featurizer=featurizer,
                    spec_augmentation=spec_augmentation,
                    context_window=context_window,
                    window_stride=window_stride,
                    subsampling=subsampling,
                    train_segment_seconds=train_segment_seconds,
                    max_speakers=max_speakers,
                )
            )
        datasets.append(
            VoxCelebDataset(
                preprocessor=preprocessor,
                featurizer=featurizer,
                spec_augmentation=spec_augmentation,
                context_window=context_window,
                window_stride=window_stride,
                subsampling=subsampling,
                config=voxceleb_config,
                max_speakers=max_speakers,
                train_segment_seconds=train_segment_seconds,
            )
        )
        return [ConcatDataset(datasets, batch_size=split_size, weights=weights) for _ in range(num_workers)]
