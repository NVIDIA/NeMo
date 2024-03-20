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

import torch


@dataclass
class DataItem:
    sample_id: str | None = None


@dataclass
class DataBatch:
    sample_id: list | None = None


@dataclass
class AudioItem(DataItem):
    audio: torch.Tensor | None = None
    audio_len: torch.Tensor | None = None


@dataclass
class AudioBatch(DataBatch):
    audio: torch.Tensor | None = None
    audio_len: torch.Tensor | None = None


@dataclass
class AudioTextItem(AudioItem):
    text: torch.Tensor | None = None
    text_len: torch.Tensor | None = None


@dataclass
class AudioTextBatch(AudioBatch):
    text: torch.Tensor | None = None
    text_len: torch.Tensor | None = None


@dataclass
class AudioNoiseItem(AudioItem):
    sample_id: str | None = None
    noise: torch.Tensor | None = None
    noise_len: torch.Tensor | None = None
    noisy_audio: torch.Tensor | None = None
    noisy_audio_len: torch.Tensor | None = None


@dataclass
class AudioNoiseBatch(AudioBatch):
    sample_id: list | None = None
    noise: torch.Tensor | None = None
    noise_len: torch.Tensor | None = None
    noisy_audio: torch.Tensor | None = None
    noisy_audio_len: torch.Tensor | None = None
