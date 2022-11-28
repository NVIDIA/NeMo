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
import glob
import math
import os
import random
from dataclasses import dataclass
from itertools import chain, cycle
from typing import Iterable, List, Optional

import numpy as np
import sox
import torch
import torch.nn.functional as F
from scipy.stats import halfnorm
from torch.utils.data import IterableDataset

from nemo.collections.asr.data.data_simulation import clamp_min_list
from nemo.collections.asr.data.deep_diarize.utils import ContextWindow, assign_frame_level_spk_vector
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.modules.audio_preprocessing import SpectrogramAugmentation
from nemo.collections.asr.parts.preprocessing import WaveformFeaturizer


@dataclass
class VoxCelebConfig:
    voxceleb_path: Optional[str]
    max_call_length_seconds: int
    dominance_var: float = 0.11
    min_dominance: float = 0.05
    turn_prob: float = 0.875
    mean_overlap: float = 0.19
    mean_silence: float = 0.15
    overlap_prob: float = 0.5
    end_buffer: float = 0.5


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


class VoxCelebDataset(IterableDataset):
    def __init__(
        self,
        preprocessor: AudioToMelSpectrogramPreprocessor,
        featurizer: WaveformFeaturizer,
        spec_augmentation: SpectrogramAugmentation,
        context_window: ContextWindow,
        window_stride: float,
        subsampling: int,
        max_speakers: int,
        config: VoxCelebConfig,
        train_segment_seconds: int,
    ):
        self.subsampling = subsampling
        self.train_segment_seconds = train_segment_seconds
        self.preprocessor = preprocessor
        self.featurizer = featurizer
        self.spec_augmentation = spec_augmentation
        self.context_window = context_window
        self.sample_rate = self.featurizer.sample_rate
        self.max_sequential_segments = int(config.max_call_length_seconds / train_segment_seconds)
        self.round_digits = 2
        self.max_speakers = max_speakers
        self.frame_per_sec = int(1 / (window_stride * subsampling))
        self.dominance_var = config.dominance_var
        self.min_dominance = config.min_dominance
        self.turn_prob = config.turn_prob
        self.mean_overlap = config.mean_overlap
        self.mean_silence = config.mean_silence
        self.overlap_prob = config.overlap_prob
        self.end_buffer = config.end_buffer
        self.voxceleb_path = config.voxceleb_path
        self.voxceleb = {}
        for speaker_id in os.listdir(config.voxceleb_path):
            files = os.path.join(config.voxceleb_path, speaker_id)
            wav_files = [os.path.join(files, wav) for wav in glob.glob(files + '/**/*.wav')]
            self.voxceleb[speaker_id] = wav_files

        self.speakers = list(self.voxceleb.keys())

    def get_stream(self):
        return chain.from_iterable(map(self.process_data, cycle([None])))

    def __iter__(self):
        return self.get_stream()

    def _get_speaker_dominance(self, num_speakers: int) -> List[float]:
        """
        Get the dominance value for each speaker, accounting for the dominance variance and
        the minimum per-speaker dominance.

        Returns:
            dominance (list): Per-speaker dominance
        """
        dominance_mean = 1.0 / num_speakers
        dominance = np.random.normal(loc=dominance_mean, scale=self.dominance_var, size=num_speakers,)
        dominance = clamp_min_list(dominance, 0)
        # normalize while maintaining minimum dominance
        total = np.sum(dominance)
        if total == 0:
            for i in range(len(dominance)):
                dominance[i] += self.min_dominance
        # scale accounting for min_dominance which has to be added after
        dominance = (dominance / total) * (1 - self.min_dominance * num_speakers)
        for i in range(len(dominance)):
            dominance[i] += self.min_dominance
            if (
                i > 0
            ):  # dominance values are cumulative to make it easy to select the speaker using a random value in [0,1]
                dominance[i] = dominance[i] + dominance[i - 1]
        return dominance

    def decision(self, probability):
        return random.random() < probability

    def _get_next_speaker(self, prev_speaker: int, dominance: List[float]) -> int:
        """
        Get the next speaker (accounting for turn probability and dominance distribution).

        Args:
            prev_speaker (int): Previous speaker turn.
            dominance (list): Dominance values for each speaker.
        Returns:
            prev_speaker/speaker_turn (int): Speaker turn
        """
        if np.random.uniform(0, 1) > self.turn_prob and prev_speaker is not None:
            return prev_speaker
        else:
            speaker_turn = prev_speaker
            while speaker_turn == prev_speaker:  # ensure another speaker goes next
                rand = np.random.uniform(0, 1)
                speaker_turn = 0
                while rand > dominance[speaker_turn]:
                    speaker_turn += 1
            return speaker_turn

    def _add_silence_or_overlap(
        self,
        speaker_turn: int,
        prev_speaker: int,
        start: int,
        length: int,
        session_length_sr: int,
        prev_length_sr: int,
        furthest_sample,
        missing_overlap,
    ):
        """
        Returns new overlapped (or shifted) start position after inserting overlap or silence.

        Args:
            speaker_turn (int): Current speaker turn.
            prev_speaker (int): Previous speaker turn.
            start (int): Current start of the audio file being inserted.
            length (int): Length of the audio file being inserted.
            session_length_sr (int): Running length of the session in terms of number of samples
            prev_length_sr (int): Length of previous sentence (in terms of number of samples)
        """
        op = self.overlap_prob / self.turn_prob  # accounting for not overlapping the same speaker
        mean_overlap_percent = (self.mean_overlap / (1 + self.mean_overlap)) / self.overlap_prob
        mean_silence_percent = self.mean_silence / (1 - self.overlap_prob)

        # overlap
        if prev_speaker != speaker_turn and prev_speaker is not None and np.random.uniform(0, 1) < op:
            overlap_percent = halfnorm(loc=0, scale=mean_overlap_percent * np.sqrt(np.pi) / np.sqrt(2)).rvs()
            desired_overlap_amount = int(prev_length_sr * overlap_percent)
            new_start = start - desired_overlap_amount

            # reinject missing overlap to ensure desired overlap percentage is met
            if missing_overlap > 0 and overlap_percent < 1:
                rand = int(prev_length_sr * np.random.uniform(0, 1 - overlap_percent / (1 + self.mean_overlap)))
                if rand > missing_overlap:
                    new_start -= missing_overlap
                    desired_overlap_amount += missing_overlap
                    missing_overlap = 0
                else:
                    new_start -= rand
                    desired_overlap_amount += rand
                    missing_overlap -= rand

            # avoid overlap at start of clip
            if new_start < 0:
                desired_overlap_amount -= 0 - new_start
                missing_overlap += 0 - new_start
                new_start = 0

            # if same speaker ends up overlapping from any previous clip, pad with silence instead
            if new_start < furthest_sample[speaker_turn]:
                desired_overlap_amount -= furthest_sample[speaker_turn] - new_start
                missing_overlap += furthest_sample[speaker_turn] - new_start
                new_start = furthest_sample[speaker_turn]

            prev_start = start - prev_length_sr
            prev_end = start
            new_end = new_start + length
            overlap_amount = 0
            if prev_start < new_start and new_end > prev_end:
                overlap_amount = prev_end - new_start
            elif prev_start < new_start and new_end < prev_end:
                overlap_amount = new_end - new_start
            elif prev_start > new_start and new_end < prev_end:
                overlap_amount = new_end - prev_start
            elif prev_start > new_start and new_end > prev_end:
                overlap_amount = prev_end - prev_start

            overlap_amount = max(overlap_amount, 0)
            if overlap_amount < desired_overlap_amount:
                missing_overlap += desired_overlap_amount - overlap_amount

        else:
            # add silence
            silence_percent = halfnorm(loc=0, scale=mean_silence_percent * np.sqrt(np.pi) / np.sqrt(2)).rvs()
            silence_amount = int(length * silence_percent)

            if start + length + silence_amount > session_length_sr:
                # don't add silence
                new_start = start
            else:
                new_start = start + silence_amount
        return new_start, furthest_sample, missing_overlap

    @property
    def num_sequential_segments(self):
        return random.randint(1, self.max_sequential_segments)

    def parse_rttm_for_ms_targets(
        self, rttm_timestamps: Iterable, offset: float, end_duration: float, speakers: List,
    ):
        fr_level_target = assign_frame_level_spk_vector(
            rttm_timestamps=rttm_timestamps,
            round_digits=self.round_digits,
            frame_per_sec=self.frame_per_sec,
            subsampling=self.subsampling,
            preprocessor=self.preprocessor,
            sample_rate=self.sample_rate,
            start_duration=offset,
            end_duration=end_duration,
            speakers=speakers,
        )
        return fr_level_target

    def process_data(self, _):
        num_speakers = random.randint(2, self.max_speakers)
        speakers = random.sample(self.speakers, num_speakers)
        speaker_dominance = self._get_speaker_dominance(num_speakers)  # randomly determine speaker dominance

        start_segment = True

        for x in range(self.num_sequential_segments):
            train_segment, rttm_timestamps = self.generate(speakers, speaker_dominance, num_speakers)
            stt_list, end_list, speaker_list = rttm_timestamps
            target_speakers = sorted(list(set(speaker_list)))

            targets = self.parse_rttm_for_ms_targets(
                rttm_timestamps=rttm_timestamps,
                offset=0,
                end_duration=self.train_segment_seconds,
                speakers=target_speakers,
            )

            train_length = torch.tensor(train_segment.shape[0]).long()

            train_segment, train_length = self.preprocessor.get_features(
                train_segment.unsqueeze_(0), train_length.unsqueeze_(0)
            )

            train_segment = self.context_window(train_segment.transpose(1, 2).squeeze(0)).unsqueeze(0)
            train_segment = train_segment.transpose(1, 2)
            train_segment = self.spec_augmentation(input_spec=train_segment, length=train_length)
            # pad targets to max speakers
            targets = F.pad(targets, pad=(0, self.max_speakers - targets.size(-1)))
            yield train_segment, train_length, targets, start_segment
            start_segment = False

    def generate(self, speakers: List[str], speaker_dominance: List[float], num_speakers: int):
        # randomly select speakers, a background noise and a SNR
        speaker_timelines = {}

        session_length_sr = self.train_segment_seconds * self.sample_rate

        prev_speaker = None
        running_length_sr, prev_length_sr = 0, 0
        furthest_sample = [0 for n in range(num_speakers)]
        missing_overlap = 0

        while running_length_sr < session_length_sr:
            # select speaker
            speaker_turn = self._get_next_speaker(prev_speaker, speaker_dominance)

            # build sentence (only add if remaining length >  specific time)
            max_sentence_duration_sr = session_length_sr - running_length_sr

            if max_sentence_duration_sr < self.end_buffer * self.sample_rate:
                break

            utterance = random.choice(self.voxceleb[speakers[speaker_turn]])
            duration = sox.file_info.duration(utterance)

            length = int(duration * self.sample_rate)
            start, furthest_sample, missing_overlap = self._add_silence_or_overlap(
                speaker_turn=speaker_turn,
                prev_speaker=prev_speaker,
                start=running_length_sr,
                length=length,
                session_length_sr=session_length_sr,
                prev_length_sr=prev_length_sr,
                furthest_sample=furthest_sample,
                missing_overlap=missing_overlap,
            )
            end = start + length

            speaker_timelines[speaker_turn] = speaker_timelines.get(speaker_turn, [])
            speaker_timelines[speaker_turn].append(
                {'utt': utterance, 'length': duration, 'offset': 0, 'total_offset': start}
            )

            running_length_sr = np.maximum(running_length_sr, end)
            furthest_sample[speaker_turn] = running_length_sr
            prev_speaker = speaker_turn
            prev_length_sr = length

        max_size = math.ceil(
            max(
                max(sample['total_offset'] + (sample['length'] * self.sample_rate) for sample in timeline)
                for k, timeline in speaker_timelines.items()
            )
        )
        combined = []
        stt_list, end_list, speaker_list = [], [], []
        for speaker_id, key in enumerate(speaker_timelines.keys()):
            data = np.zeros(max_size)
            for sample in speaker_timelines[key]:
                start_offset = int(sample['total_offset'])
                speech = self.featurizer.process(
                    sample['utt'], offset=sample['offset'], duration=sample['length'], channel_selector='average'
                ).numpy()
                end_offset = start_offset + len(speech)
                data[start_offset:end_offset] = speech
                start_second = start_offset / self.sample_rate
                stt_list.append(start_second)
                end_list.append(start_second + sample['length'])
                speaker_list.append(speaker_id)
            combined.append(data)

        # fitting to the maximum-length speaker data, then mix all speakers
        maxlen = max(len(x) for x in combined)
        combined = [np.pad(x, (0, maxlen - len(x)), 'constant') for x in combined]
        combined = np.sum(combined, axis=0)
        rttm_timestamps = (stt_list, end_list, speaker_list)
        return torch.tensor(combined, dtype=torch.float32), rttm_timestamps
