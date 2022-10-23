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

import torch
import webdataset as wds

from nemo.collections.asr.data.audio_to_diar_label import extract_seg_info_from_rttm
from nemo.collections.asr.data.audio_to_text import expand_audio_filepaths
from nemo.collections.asr.data.deep_diarize.train_data import RTTMStreamingSegmentsDataset
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.parts.preprocessing import WaveformFeaturizer


class TarredRTTMStreamingSegmentsDataset(RTTMStreamingSegmentsDataset):
    def process_data(self, x):
        start_segment = True
        audio_bytes, rttm_file = x['wav'], x['rttm']

        rttm_filestream = io.BytesIO(rttm_file)
        rttm_lines = rttm_filestream.readlines()

        # todo: unique ID isn't needed
        rttm_timestamps = extract_seg_info_from_rttm("", rttm_lines)
        stt_list, end_list, speaker_list = rttm_timestamps
        total_annotated_duration = max(end_list)
        n_segments = math.ceil(total_annotated_duration / self.train_segment_seconds)
        start_offset = 0
        for n_segment in range(n_segments):
            duration = self.train_segment_seconds

            targets = self.parse_rttm_for_ms_targets(
                rttm_timestamps, offset=start_offset, end_duration=start_offset + duration,
            )
            # Convert audio bytes to IO stream for processing (for SoundFile to read)
            audio_filestream = io.BytesIO(audio_bytes)
            train_segment = self.featurizer.process(audio_filestream, offset=start_offset, duration=duration)

            train_length = torch.tensor(train_segment.shape[0]).long()

            train_segment, train_length = self.preprocessor.get_features(
                train_segment.unsqueeze_(0), train_length.unsqueeze_(0)
            )
            start_offset += duration
            yield train_segment, train_length, targets, start_segment
            start_segment = False

    def get_streams(self):
        url = expand_audio_filepaths(self.manifest_filepath, shard_strategy='replicate', world_size=1, global_rank=1,)
        worker_seed = torch.initial_seed() % 2 ** 32
        dataset = wds.WebDataset(url).shuffle(100, rng=random.Random(worker_seed))
        return zip(*[self.get_stream(iter(dataset)) for _ in range(self.batch_size)])

    @classmethod
    def create_datasets(
        cls,
        featurizer: WaveformFeaturizer,
        manifest_filepath: str,
        num_workers: int,
        preprocessor: AudioToMelSpectrogramPreprocessor,
        split_size: int,
        subsampling: int,
        train_segment_seconds: int,
        window_stride: float,
    ):
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
