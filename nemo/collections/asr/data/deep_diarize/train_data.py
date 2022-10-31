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

import math
import random
from abc import ABC
from itertools import chain, cycle
from typing import List

import torch
from torch.utils.data import DataLoader, IterableDataset

from nemo.collections.asr.data.audio_to_diar_label import extract_seg_info_from_rttm
from nemo.collections.asr.data.deep_diarize.utils import assign_frame_level_spk_vector
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.modules.audio_preprocessing import SpectrogramAugmentation
from nemo.collections.asr.parts.preprocessing import WaveformFeaturizer
from nemo.collections.common.parts.preprocessing.collections import DiarizationSpeechLabel


def _train_collate_fn(batch):
    """
    Collate batch of variables that are needed for raw waveform to diarization label training.
    The following variables are included in training/validation batch:

    Args:
        batch (tuple):
            Batch tuple containing the variables for the diarization training.
    Returns:
        features (torch.tensor):
            Raw waveform samples (time series) loaded from the audio_filepath in the input manifest file.
        feature lengths (time series sample length):
            A list of lengths of the raw waveform samples.
        targets (torch.tensor):
            Groundtruth Speaker label for the given input embedding sequence.
    """
    packed_batch = list(zip(*batch))
    (train_segment, train_length, targets, start_segment,) = packed_batch

    train_segments = torch.cat(train_segment).transpose(1, 2)
    train_segments_lengths = torch.stack(train_length)
    targets = torch.stack(targets)
    return (
        train_segments,
        train_segments_lengths,
        targets,
        start_segment,
    )


class RTTMStreamingSegmentsDataset(IterableDataset, ABC):
    def __init__(
        self,
        batch_size,
        manifest_filepath: str,
        preprocessor: AudioToMelSpectrogramPreprocessor,
        featurizer: WaveformFeaturizer,
        spec_augmentation: SpectrogramAugmentation,
        window_stride: float,
        subsampling: int,
        train_segment_seconds: int,
    ):
        self.batch_size = batch_size
        self.subsampling = subsampling
        self.train_segment_seconds = train_segment_seconds
        self.preprocessor = preprocessor
        self.featurizer = featurizer
        self.spec_augmentation = spec_augmentation
        self.round_digits = 2
        self.max_spks = 2
        self.frame_per_sec = int(1 / (window_stride * subsampling))
        self.manifest_filepath = manifest_filepath

    def parse_rttm_for_ms_targets(
        self, rttm_timestamps: list, offset: float, end_duration: float, speakers: List,
    ):
        """
        Generate target tensor variable by extracting groundtruth diarization labels from an RTTM file.
        This function converts (start, end, speaker_id) format into base-scale (the finest scale) segment level
        diarization label in a matrix form.

        Example of seg_target:
            [[0., 1.], [0., 1.], [1., 1.], [1., 0.], [1., 0.], ..., [0., 1.]]

        Args:
            sample:
                `DiarizationSpeechLabel` instance containing sample information such as audio filepath and RTTM filepath.
            target_spks (tuple):
                Speaker indices that are generated from combinations. If there are only one or two speakers,
                only a single target_spks tuple is generated.

        Returns:
            fr_level_target  (torch.tensor):
                Tensor variable containing hard-labels of speaker activity in each base-scale segment.
        """
        fr_level_target = assign_frame_level_spk_vector(
            rttm_timestamps=rttm_timestamps,
            round_digits=self.round_digits,
            frame_per_sec=self.frame_per_sec,
            subsampling=self.subsampling,
            preprocessor=self.preprocessor,
            sample_rate=self.preprocessor._sample_rate,
            start_duration=offset,
            end_duration=end_duration,
            speakers=speakers,
        )
        return fr_level_target

    @property
    def shuffled_batch_list(self):
        raise NotImplementedError

    def process_data(self, data):
        raise NotImplementedError

    def get_stream(self, data_list):
        return chain.from_iterable(map(self.process_data, cycle(data_list)))

    def get_streams(self):
        return zip(*[self.get_stream(self.shuffled_batch_list) for _ in range(self.batch_size)])

    def __iter__(self):
        return self.get_streams()

    @classmethod
    def create_streaming_datasets(
        cls,
        batch_size,
        manifest_filepath: str,
        preprocessor: AudioToMelSpectrogramPreprocessor,
        featurizer: WaveformFeaturizer,
        spec_augmentation: SpectrogramAugmentation,
        window_stride: float,
        subsampling: int,
        train_segment_seconds: int,
        max_workers,
    ):
        num_workers = max_workers
        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers = n
                break
        split_size = batch_size // num_workers
        return cls.create_datasets(
            featurizer=featurizer,
            manifest_filepath=manifest_filepath,
            num_workers=num_workers,
            preprocessor=preprocessor,
            spec_augmentation=spec_augmentation,
            split_size=split_size,
            subsampling=subsampling,
            train_segment_seconds=train_segment_seconds,
            window_stride=window_stride,
        )

    @classmethod
    def create_datasets(
        cls,
        featurizer: WaveformFeaturizer,
        manifest_filepath: str,
        num_workers: int,
        preprocessor: AudioToMelSpectrogramPreprocessor,
        spec_augmentation: SpectrogramAugmentation,
        split_size: int,
        subsampling: int,
        train_segment_seconds: int,
        window_stride: float,
    ):
        raise NotImplementedError

    @property
    def dataloader_class(self):
        return DataLoader


class LocalRTTMStreamingSegmentsDataset(RTTMStreamingSegmentsDataset):

    minimum_segment_seconds: int = 1

    def __init__(
        self,
        data_list: list,
        collection: DiarizationSpeechLabel,
        batch_size: int,
        manifest_filepath: str,
        preprocessor: AudioToMelSpectrogramPreprocessor,
        featurizer: WaveformFeaturizer,
        spec_augmentation: SpectrogramAugmentation,
        window_stride: float,
        subsampling: int,
        train_segment_seconds: int,
    ):
        super().__init__(
            batch_size,
            manifest_filepath,
            preprocessor,
            featurizer,
            spec_augmentation,
            window_stride,
            subsampling,
            train_segment_seconds,
        )
        self.data_list = data_list
        self.collection = collection

    @property
    def shuffled_batch_list(self):
        return random.sample(self.data_list, len(self.data_list))

    def process_data(self, data):
        sample, rttm_timestamps = data
        start_segment = True
        stt_list, end_list, speaker_list = rttm_timestamps
        speakers = sorted(list(set(speaker_list)))
        random.shuffle(speakers)

        total_annotated_duration = max(end_list)
        n_segments = math.ceil((total_annotated_duration - sample.offset) / self.train_segment_seconds)
        start_offset = sample.offset
        for n_segment in range(n_segments):
            duration = self.train_segment_seconds
            start_offset += duration

            if (total_annotated_duration - start_offset) > self.minimum_segment_seconds:

                targets = self.parse_rttm_for_ms_targets(
                    rttm_timestamps=rttm_timestamps,
                    offset=start_offset,
                    end_duration=start_offset + duration,
                    speakers=speakers,
                )
                train_segment = self.featurizer.process(sample.audio_file, offset=start_offset, duration=duration)

                train_length = torch.tensor(train_segment.shape[0]).long()

                train_segment, train_length = self.preprocessor.get_features(
                    train_segment.unsqueeze_(0), train_length.unsqueeze_(0)
                )

                train_segment = self.spec_augmentation(input_spec=train_segment, length=train_length)

                yield train_segment, train_length, targets, start_segment
                start_segment = False

    @staticmethod
    def data_setup(manifest_filepath: str):
        collection = DiarizationSpeechLabel(
            manifests_files=manifest_filepath.split(","), emb_dict=None, clus_label_dict=None,
        )
        samples = []
        for sample_id, sample in enumerate(collection):
            if sample.offset is None:
                sample.offset = 0

            with open(sample.rttm_file) as f:
                rttm_lines = f.readlines()
            # todo: unique ID isn't needed
            rttm_timestamps = extract_seg_info_from_rttm("", rttm_lines)
            if sample is None or rttm_timestamps is None or not rttm_timestamps:
                raise ValueError
            samples.append((sample, rttm_timestamps,))
        return collection, samples

    @classmethod
    def create_datasets(
        cls,
        featurizer: WaveformFeaturizer,
        manifest_filepath: str,
        num_workers: int,
        preprocessor: AudioToMelSpectrogramPreprocessor,
        spec_augmentation: SpectrogramAugmentation,
        split_size: int,
        subsampling: int,
        train_segment_seconds: int,
        window_stride: float,
    ):
        (collection, calls,) = cls.data_setup(manifest_filepath=manifest_filepath)
        return [
            cls(
                data_list=calls,
                collection=collection,
                manifest_filepath=manifest_filepath,
                preprocessor=preprocessor,
                featurizer=featurizer,
                spec_augmentation=spec_augmentation,
                window_stride=window_stride,
                subsampling=subsampling,
                train_segment_seconds=train_segment_seconds,
                batch_size=split_size,
            )
            for _ in range(num_workers)
        ]


class MultiStreamDataLoader:
    def __init__(self, datasets):
        self.datasets = datasets

    def get_stream_loaders(self):
        return zip(*[dataset.dataloader_class(dataset, num_workers=1, batch_size=None) for dataset in self.datasets])

    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            yield _train_collate_fn(tuple(chain(*batch_parts)))
