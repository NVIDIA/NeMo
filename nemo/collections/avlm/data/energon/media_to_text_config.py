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
from typing import Optional

import torch

from megatron.energon import Sample
from megatron.energon.flavors.webdataset import VideoData

from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.vlm.llava_next.data.energon import LlavaNextTextRawBatch, LlavaNextTextSample
from nemo.utils import logging

@dataclass_slot
class MediaToTextEnergonSample(Sample):
    context: str
    answers: Optional[List[str]] = None
    answer_weights: Optional[torch.Tensor] = None
    audio: Optional[torch.tensor] = None
    video: Optional[VideoData] = None    
    offset: Optional[float] = None
    duration: Optional[float] = None
    image: Optional[torch.tensor] = None


@dataclass
class MediaToTextSample(LlavaNextTextSample):
    '''
    Sample type for media to text task, extending LlavaNextTextSample to support audio and video data.

    This class adds additional attributes for handling the audio and video raw bytes,
    along with metadata about the tiled images.

    Attributes:
    '''
    audio: Optional[torch.tensor] = None
    video: Optional[torch.tensor] = None
    images: Optional[torch.tensor] = None
    num_media_tiles: Optional[int] = None
    image_sizes: Optional[torch.tensor] = None


@dataclass
class MediaToTextRawBatch(LlavaNextTextRawBatch):
    """
    Batch type for raw media to text samples, supporting audio, image(s).

    This class aggregates multiple `MediaToTextSample` instances into a batch for processing.
    It includes attributes for managing audio, image and associated metadata for each sample in the batch.

    Attributes:
    """
    audio: Optional[torch.tensor] = None
    video: Optional[torch.tensor] = None
    images: Optional[torch.tensor] = None
    num_media_tiles: Optional[int] = None
    image_sizes: Optional[torch.tensor] = None

@dataclass
class MediaToTextSampleConfig(MultiModalSampleConfig):
    audio_original_sampling_rate: Optional[int] = None
    concat_video_following_images: Optional[bool] = None
    