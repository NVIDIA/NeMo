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

from PIL import Image
from dataclasses import dataclass, field
from typing import Optional, List, Union

import torch

from megatron.energon import Sample
from megatron.energon.flavors.webdataset import VideoData

from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.vlm.llava_next.data.energon import LlavaNextTextRawBatch, LlavaNextTextSample
from nemo.utils import logging

@dataclass_slot
class MediaToMediaEnergonSample(Sample):
    # text related attributes
    ## contexts/questions/answers can be list of interleaved sequences
    contexts: Optional[List[List[Union[bytes, str, Image.Image, dict]]]] = None
    questions: Optional[List[List[Union[bytes, str, Image.Image, dict]]]] = None
    texts: Optional[[str]] = None
    answers: Optional[List[List[Union[bytes, str, Image.Image, dict]]]] = None
    answer_weights: Optional[torch.Tensor] = None

    # audio related attributes
    ## sample loader loads the audio raw bytes which will be decoded 
    ## and further processed in the task encoder
    audios: Optional[[bytes]] = None
    audio_offsets: Optional[[float]] = None
    audio_durations: Optional[[float]] = None

    # video related attributes
    ## sample loader loads the video raw bytes which will be decoded 
    ## and further processed in the task encoder
    videos: Optional[[bytes]] = None
    video_offsets: Optional[[float]] = None
    video_durations: Optional[[float]] = None

    # image related attributes
    ## sample loader loads and decodes the images as PIL.Image
    images: Optional[[Image.Image]] = None


@dataclass
class MediaToMediaSample:
    '''
    Sample type for media to text task, extending LlavaNextTextSample to support audio and video data.

    This class adds additional attributes for handling the audio and video raw bytes,
    along with metadata about the tiled images.

    Attributes:
    '''

    __key__: str = ''
    tokens: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    labels: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    loss_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))
    audios: Optional[torch.tensor] = None
    audio_lengths: Optional[torch.tensor] = None
    videos: Optional[torch.tensor] = None
    video_lengths: Optional[torch.tensor] = None
    num_video_tiles: Optional[torch.tensor] = None
    images: Optional[torch.tensor] = None
    num_image_tiles: Optional[int] = None
    image_sizes: Optional[torch.tensor] = None
    attention_mask: Optional[torch.tensor] = None


@dataclass
class MediaToMediaRawBatch:
    """
    Batch type for raw media to text samples, supporting audio, image(s).

    This class aggregates multiple `MediaToMediaSample` instances into a batch for processing.
    It includes attributes for managing audio, image and associated metadata for each sample in the batch.

    Attributes:
    """
    __keys__: List[str] = field(default_factory=list)
    tokens: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    labels: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    loss_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))
    audios: Optional[torch.tensor] = None
    audio_lengths: Optional[torch.tensor] = None
    videos: Optional[torch.tensor] = None
    video_lengths: Optional[torch.tensor] = None
    num_video_tiles: Optional[torch.tensor] = None
    images: Optional[torch.tensor] = None
    num_image_tiles: Optional[int] = None
    image_sizes: Optional[torch.tensor] = None
    attention_mask: Optional[torch.tensor] = None
    