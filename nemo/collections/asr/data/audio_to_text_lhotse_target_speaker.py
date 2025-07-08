# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import math
import os
import re
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.utils.data
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment, SupervisionSet
from lhotse.cut import MixedCut, MonoCut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_matrices, collate_vectors
from lhotse.utils import compute_num_samples, uuid4

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    find_segments_from_rttm,
    get_hidden_length_from_sample_length,
)
from nemo.collections.asr.parts.utils.asr_tgtspeaker_utils import (
    codec_augment,
    get_query_cut,
    get_separator_audio,
    mix_noise,
    rir_augment,
    speaker_to_target_w_query,
)
from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


class LhotseSpeechToTextTgtSpkBpeDataset(torch.utils.data.Dataset):
    """
    This dataset is based on BPE datasets from audio_to_text_lhotse.py. It reads cut object and return audio, text and speaker target tensor.
    Output follows format of query_cut - separater_audio - target-cut
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'audio_signal_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'spk_tar_id': NeuralType(('B', 'T'), LabelsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.cfg = cfg
        self.spk_tar_all_zero = self.cfg.get('spk_tar_all_zero', False)
        self.num_speakers = self.cfg.get('num_speakers', 4)
        self.num_sample_per_mel_frame = self.cfg.get('num_sample_per_mel_frame', 160)
        self.num_mel_frame_per_asr_frame = self.cfg.get('num_mel_frame_per_asr_frame', 8)
        self.separater_freq = self.cfg.get('separater_freq', 500)
        self.separater_duration = self.cfg.get('separater_duration', 1)
        self.separater_unvoice_ratio = self.cfg.get('separater_unvoice_ratio', 0.3)
        self.separater_audio = get_separator_audio(
            self.separater_freq, self.cfg.sample_rate, self.separater_duration, self.separater_unvoice_ratio
        )
        self.add_special_token = self.cfg.get('add_special_token', True)
        if self.add_special_token:
            self.special_token = self.cfg.get('special_token', '<|beep|>')
        # augmentation config
        self.query_noise_augment = self.cfg.get('query_noise_augment', False)
        self.query_rir_augment = self.cfg.get('query_rir_augment', False)
        self.query_rir_prob = self.cfg.get('query_rir_prob', 0.3)
        self.query_codec_augment = self.cfg.get('query_codec_augment', False)
        self.query_codec_prob = self.cfg.get('query_codec_prob', 0.3)
        if self.query_noise_augment:
            self.query_noise_path = self.cfg.get('query_noise_path', None)
            if not self.query_noise_path:
                raise ValueError('query_noise_path is not set')
            with open(self.query_noise_path, 'r') as f:
                self.query_noise_manifests = [json.loads(line) for line in f]
            self.query_noise_mix_prob = self.cfg.get('query_noise_mix_prob', 0.3)
            self.query_snr = tuple(self.cfg.get('query_snr', (2.5, 12.5)))

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:

        query_cuts = CutSet.from_cuts(get_query_cut(c) for c in cuts)

        spk_targets = [
            torch.transpose(
                torch.as_tensor(
                    speaker_to_target_w_query(
                        c,
                        q,
                        self.separater_duration,
                        self.num_speakers,
                        self.num_sample_per_mel_frame,
                        self.num_mel_frame_per_asr_frame,
                        self.spk_tar_all_zero,
                    ),
                    dtype=torch.float32,
                ),
                0,
                1,
            )
            for c, q in zip(cuts, query_cuts)
        ]

        # order matters: rir_augment and codec_augment (output monocut) should be applied before mix_noise (output mixedcut)
        if self.query_rir_augment:
            query_cuts = rir_augment(query_cuts, prob=self.query_rir_prob)
        if self.query_codec_augment:
            query_cuts = codec_augment(query_cuts, prob=self.query_codec_prob)

        if self.query_noise_augment:
            query_cuts = mix_noise(
                query_cuts,
                self.query_noise_manifests,
                snr=self.query_snr,
                mix_prob=self.query_noise_mix_prob,
            )

        audio, audio_lens, cuts = self.load_audio(cuts)
        query_audio, query_audio_lens, query_cuts = self.load_audio(query_cuts)
        # concate query audio, separater audio and target audio
        concat_list = []
        for i in range(len(audio)):
            concat_list.append(
                torch.cat(
                    [
                        query_audio[i, : query_audio_lens[i]],
                        torch.tensor(self.separater_audio).to(audio.dtype),
                        audio[i, : audio_lens[i]],
                    ]
                )
            )
        audio = collate_vectors(concat_list, padding_value=0)
        audio_lens = audio_lens + query_audio_lens + self.separater_duration * self.cfg.sample_rate
        if self.add_special_token:
            tokens = [
                torch.as_tensor(
                    self.tokenizer(self.special_token + ' ' + c.supervisions[0].text, c.supervisions[0].language)
                )
                for c in cuts
            ]
        else:
            tokens = [
                torch.as_tensor(self.tokenizer(c.supervisions[0].text, c.supervisions[0].language)) for c in cuts
            ]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)
        spk_targets = collate_matrices(spk_targets)
        return audio, audio_lens, tokens, token_lens, spk_targets
