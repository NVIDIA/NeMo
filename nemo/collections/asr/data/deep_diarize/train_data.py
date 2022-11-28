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
from itertools import chain, cycle
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from nemo.collections.asr.data.audio_to_diar_label import extract_seg_info_from_rttm
from nemo.collections.asr.data.deep_diarize.utils import ContextWindow, assign_frame_level_spk_vector
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


class LocalRTTMStreamingSegmentsDataset(IterableDataset):
    minimum_segment_seconds: int = 1

    def __init__(
        self,
        data_list: list,
        manifest_filepath: str,
        preprocessor: AudioToMelSpectrogramPreprocessor,
        featurizer: WaveformFeaturizer,
        spec_augmentation: SpectrogramAugmentation,
        context_window: ContextWindow,
        window_stride: float,
        subsampling: int,
        train_segment_seconds: int,
        max_speakers: int,
    ):
        self.data_list = data_list
        self.max_speakers = max_speakers
        self.subsampling = subsampling
        self.train_segment_seconds = train_segment_seconds
        self.preprocessor = preprocessor
        self.featurizer = featurizer
        self.spec_augmentation = spec_augmentation
        self.context_window = context_window
        self.round_digits = 2
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

    def get_stream(self, data_list):
        return chain.from_iterable(map(self.process_data, cycle(data_list)))

    def __iter__(self):
        return self.get_stream(self.shuffled_batch_list)

    @property
    def shuffled_batch_list(self):
        return random.sample(self.data_list, len(self.data_list))

    def process_data(self, data):
        sample, rttm_timestamps, speakers = data
        start_segment = True
        stt_list, end_list, speaker_list = rttm_timestamps
        random.shuffle(speakers)

        total_annotated_duration = max(end_list)
        n_segments = math.ceil((total_annotated_duration - sample.offset) / self.train_segment_seconds)
        start_offset = sample.offset
        try:
            for n_segment in range(n_segments):
                duration = self.train_segment_seconds

                if (total_annotated_duration - start_offset) > self.minimum_segment_seconds:
                    targets = self.parse_rttm_for_ms_targets(
                        rttm_timestamps=rttm_timestamps,
                        offset=start_offset,
                        end_duration=start_offset + duration,
                        speakers=speakers,
                    )
                    # pad targets to max speakers
                    targets = F.pad(targets, pad=(0, self.max_speakers - targets.size(-1)))
                    train_segment = self.featurizer.process(
                        sample.audio_file, offset=start_offset, duration=duration, channel_selector='average'
                    )

                    train_length = torch.tensor(train_segment.shape[0]).long()

                    train_segment, train_length = self.preprocessor.get_features(
                        train_segment.unsqueeze_(0), train_length.unsqueeze_(0)
                    )

                    # todo: this stacking procedure requires thought when combined with spec augment
                    train_segment = self.context_window(train_segment.transpose(1, 2).squeeze(0)).unsqueeze(0)
                    train_segment = train_segment.transpose(1, 2)
                    train_segment = self.spec_augmentation(input_spec=train_segment, length=train_length)

                    yield train_segment, train_length, targets, start_segment
                    start_segment = False
                    start_offset += duration
        except Exception as e:
            print("Failed data-loading for", sample.audio_file, e)

    @staticmethod
    def data_setup(manifest_filepath: str, max_speakers: int):
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
            stt_list, end_list, speaker_list = rttm_timestamps
            speakers = sorted(list(set(speaker_list)))
            samples.append((sample, rttm_timestamps, speakers))
        pruned_samples = []
        for sample in samples:
            _, rttm_timestamps, speakers = sample
            if len(speakers) <= max_speakers:
                pruned_samples.append(sample)
        print(f"pruned {len(samples) - len(pruned_samples)} out of {len(samples)} calls")
        return pruned_samples


class MultiStreamDataLoader:
    def __init__(self, datasets):
        self.datasets = datasets

    def get_stream_loaders(self):
        return zip(*[DataLoader(dataset, num_workers=1, batch_size=None) for dataset in self.datasets])

    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            yield _train_collate_fn(tuple(chain(*batch_parts)))
