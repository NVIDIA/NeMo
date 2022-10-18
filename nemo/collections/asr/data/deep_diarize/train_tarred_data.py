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
import io
import math
import random
from itertools import chain, cycle

import torch
import webdataset as wds
from torch.utils.data import IterableDataset

from nemo.collections.asr.data.audio_to_diar_label import extract_seg_info_from_rttm
from nemo.collections.asr.data.audio_to_text import expand_audio_filepaths
from nemo.collections.asr.data.deep_diarize.utils import assign_frame_level_spk_vector
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.parts.preprocessing import WaveformFeaturizer


class TarredRTTMStreamingSegmentsDataset(IterableDataset):
    def __init__(
        self,
        batch_size,
        manifest_filepath: str,
        preprocessor: AudioToMelSpectrogramPreprocessor,
        featurizer: WaveformFeaturizer,
        window_stride: float,
        subsampling: int,
        train_segment_seconds: int,
    ):
        self.batch_size = batch_size
        self.subsampling = subsampling
        self.train_segment_seconds = train_segment_seconds
        self.preprocessor = preprocessor
        self.featurizer = featurizer
        self.round_digits = 2
        self.max_spks = 2
        self.frame_per_sec = int(1 / (window_stride * subsampling))
        self.manifest_filepath = manifest_filepath

    def parse_rttm_for_ms_targets(
        self, rttm_timestamps, total_annotated_duration: int, offset: float, end_duration: float,
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
            total_annotated_duration=total_annotated_duration,
            round_digits=self.round_digits,
            frame_per_sec=self.frame_per_sec,
            subsampling=self.subsampling,
            preprocessor=self.preprocessor,
            sample_rate=self.preprocessor._sample_rate,
            start_duration=offset,
            end_duration=end_duration,
        )
        return fr_level_target

    def process_data(self, x):
        reset_memory = True
        audio_bytes, rttm_file = x['wav'], x['rttm']

        rttm_filestream = io.BytesIO(rttm_file)
        rttm_lines = rttm_filestream.readlines()

        # todo: unique ID isn't needed
        rttm_timestamps = extract_seg_info_from_rttm("", rttm_lines)
        stt_list, end_list, speaker_list = rttm_timestamps
        total_annotated_duration = max(end_list)
        n_segments = math.floor(total_annotated_duration / self.train_segment_seconds)
        start_offset = 0
        n_segments -= 1
        for n_segment in range(n_segments):
            duration = self.train_segment_seconds

            targets = self.parse_rttm_for_ms_targets(
                rttm_timestamps, total_annotated_duration, offset=start_offset, end_duration=start_offset + duration,
            )
            # Convert audio bytes to IO stream for processing (for SoundFile to read)
            audio_filestream = io.BytesIO(audio_bytes)
            train_segment = self.featurizer.process(audio_filestream, offset=start_offset, duration=duration)

            train_length = torch.tensor(train_segment.shape[0]).long()

            train_segment, train_length = self.preprocessor.get_features(
                train_segment.unsqueeze_(0), train_length.unsqueeze_(0)
            )
            start_offset += duration
            yield train_segment, train_length, targets, reset_memory
            reset_memory = False

    def get_stream(self, dataset):
        return chain.from_iterable(map(self.process_data, cycle(dataset)))

    def get_streams(self):
        url = expand_audio_filepaths(self.manifest_filepath, shard_strategy='replicate', world_size=1, global_rank=1,)
        worker_seed = torch.initial_seed() % 2 ** 32
        dataset = wds.WebDataset(url).shuffle(100, rng=random.Random(worker_seed))
        return zip(*[self.get_stream(iter(dataset)) for _ in range(self.batch_size)])

    def __iter__(self):
        return self.get_streams()

    @classmethod
    def create_streaming_datasets(
        cls,
        batch_size,
        manifest_filepath: str,
        preprocessor: AudioToMelSpectrogramPreprocessor,
        featurizer: WaveformFeaturizer,
        window_stride: float,
        subsampling: int,
        train_segment_seconds: int,
        max_workers: int,
        num_calls: int,
    ):
        num_workers = max_workers
        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers = n
                break
        split_size = batch_size // num_workers
        return [
            cls(
                manifest_filepath=manifest_filepath,
                preprocessor=preprocessor,
                featurizer=featurizer,
                window_stride=window_stride,
                subsampling=subsampling,
                train_segment_seconds=train_segment_seconds,
                batch_size=split_size,
            )
            for _ in range(num_workers)
        ]
