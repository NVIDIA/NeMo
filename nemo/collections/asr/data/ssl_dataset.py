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

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

from nemo.collections.asr.data import audio_to_text
from nemo.core.classes import Dataset, IterableDataset


@dataclass
class SSLAudioItem:
    sample_id: str | None = None
    audio: torch.Tensor | None = None
    audio_len: torch.Tensor | None = None
    aug_audio: torch.Tensor | None = None
    aug_audio_len: torch.Tensor | None = None
    noise_audio: torch.Tensor | None = None
    noise_audio_len: torch.Tensor | None = None


@dataclass
class SSLAudioBatch:
    sample_id: List | None = None
    audio: torch.Tensor | None = None
    audio_len: torch.Tensor | None = None
    aug_audio: torch.Tensor | None = None
    aug_audio_len: torch.Tensor | None = None


def _ssl_audio_collate_fn(batch: List[SSLAudioItem], pad_id: int = 0) -> SSLAudioBatch:
    """Collate function for SSL noisy data.

    Args:
        batch: List of tuple[audio, audio_len, text, text_len, (sample_id)] from audio_to_text datasets.

    Returns:
        AudioTextBatch: Collated batch.
    """
    packed_batch = list(zip(*batch))
    if len(packed_batch) == 5:
        _, audio_lengths, _, tokens_lengths, sample_ids = packed_batch
    elif len(packed_batch) == 4:
        sample_ids = None
        _, audio_lengths, _, tokens_lengths = packed_batch
    else:
        raise ValueError("Expects 4 or 5 tensors in the batch!")

    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    max_tokens_len = max(tokens_lengths).item()

    audio_signal, tokens = [], []
    for b in batch:
        sig, sig_len, tokens_i, tokens_i_len = b[:4]
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            pad = (0, max_tokens_len - tokens_i_len)
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)
    if sample_ids is not None:
        sample_ids = torch.tensor(sample_ids, dtype=torch.int32)


class SSLAudioDataset(audio_to_text.AudioToCharDataset):
    def __init__(
        self, noise_manifest: str | List[str] | None = None, labels: str | List[str] | None = None, **kwargs,
    ):
        super().__init__(labels=labels, **kwargs)
        self.noise_manifest = noise_manifest

    def __getitem__(self, index):
        sample = self.manifest_processor.collection[index]
        offset = sample.offset

        if offset is None:
            offset = 0

        features = self.featurizer.process(
            sample.audio_file,
            offset=offset,
            duration=sample.duration,
            trim=self.trim,
            orig_sr=sample.orig_sr,
            channel_selector=self.channel_selector,
        )
        f, fl = features, torch.tensor(features.shape[0]).long()
