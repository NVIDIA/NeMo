# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import re

import numpy as np
import torch
import torch.utils.data
from lhotse import CutSet, Recording, Seconds, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_audio, collate_vectors

from nemo.collections.common.tokenizers import TokenizerSpec


class DuplexS2SDataset(torch.utils.data.Dataset):
    """
    TODO: documentation
    """

    def __init__(self, tokenizer: TokenizerSpec, frame_length: Seconds, source_sample_rate: int):
        self.tokenizer = tokenizer
        self.frame_length = frame_length
        self.source_sample_rate = source_sample_rate

    def __getitem__(self, cuts: CutSet) -> dict:
        source_audio, source_audio_lens = collate_audio(cuts.resample(self.source_sample_rate))
        target_audio, target_audio_lens = collate_audio(cuts, recording_field="target_audio")
        target_tokens, target_token_lens = collate_token_channel(cuts, self.tokenizer, self.frame_length, role="agent")
        source_tokens, source_token_lens = collate_token_channel(cuts, self.tokenizer, self.frame_length, role="user")

        return {
            "source_audio": source_audio,
            "source_audio_lens": source_audio_lens,
            "target_audio": target_audio,
            "target_audio_lens": target_audio_lens,
            "target_tokens": target_tokens,
            "target_token_lens": target_token_lens,
            "source_tokens": source_tokens,
            "source_token_lens": source_token_lens,
        }


def collate_token_channel(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    role: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = tokenizer.pad
    if pad_id is None:
        pad_id = tokenizer.unk_id
    if pad_id is None:
        pad_id = 0  # TODO: cleanup
    tokens = [
        build_token_channel(c, tokenizer=tokenizer, frame_length=frame_length, role=role, pad_id=pad_id) for c in cuts
    ]
    token_lens = [len(tt) for tt in tokens]
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens


def build_token_channel(
    cut: Cut,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    role: str = "agent",
    pad_id: int = -1,
) -> torch.Tensor:
    bos = [] if tokenizer.bos is None else [tokenizer.bos]
    eos = [] if tokenizer.eos is None else [tokenizer.eos]
    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id
    offset = 0
    for supervision in cut.supervisions:
        d = supervision.duration
        if supervision.speaker == role:
            text_ids = torch.as_tensor(bos + tokenizer.text_to_ids(supervision.text) + eos)
            pos = compute_num_frames(offset, frame_length, cut.sampling_rate)
            # TODO: at least emit a warning if a truncation happens
            tokens[pos : pos + len(text_ids)] = text_ids
        offset += d
    return tokens
