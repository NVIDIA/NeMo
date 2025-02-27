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

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.vlm.llava_next.data.energon import LlavaNextTextRawBatch, LlavaNextTextSample
from nemo.utils import logging


@dataclass
class MediaToTextSample(LlavaNextTextSample):
    '''
    Sample type for media to text task, extending LlavaNextTextSample to support audio and video data.

    This class adds additional attributes for handling the audio and video raw bytes,
    along with metadata about the tiled images.

    Attributes:
    '''

    audio = field(default_factory=lambda: torch.tensor([]))
    # video is optional since it can be sequence of images
    video: Optional[field(default_factory=lambda: torch.tensor([]))] = None


@dataclass
class MediaToTextRawBatch(LlavaNextTextRawBatch):
    """
    Batch type for raw media to text samples, supporting audio, image(s).

    This class aggregates multiple `MediaToTextSample` instances into a batch for processing.
    It includes attributes for managing audio, image and associated metadata for each sample in the batch.

    Attributes:
    """
    audio = field(default_factory=lambda: torch.tensor([]))
    video: Optional[field(default_factory=lambda: torch.tensor([]))] = None

@dataclass
class MediaToTextSampleConfig(MultiModalSampleConfig):
    offset: float = 0
    duration: Optional[float] = 0
    audio_original_sampling_rate: Optional[int] = None
    