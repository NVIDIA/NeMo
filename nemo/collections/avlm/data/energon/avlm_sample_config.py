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
from typing import Optional, List, Union, TypedDict, NotRequired, Literal

import torch

from megatron.energon import Sample
from megatron.energon.flavors.webdataset import VideoData

from nemo.collections.multimodal.data.energon.config import AudioToken, VideoToken, MultiModalSampleConfig
from nemo.collections.vlm.llava_next.data.energon import LlavaNextTextRawBatch, LlavaNextTextSample
from nemo.utils import logging


@dataclass
class AudioSize:
    length: int
    channel: int


@dataclass
class VideoSize:
    frames: int
    height: int
    width: int


class MediaDict(TypedDict):
    media_type: Literal["audio", "video"]
    media_value: bytes
    offset: NotRequired[float]
    duration: NotRequired[float]


@dataclass_slot
class AVLMEnergonInterleavedSample(Sample):
    # sequence of interleaved media, (either PIL.Image for an image, str for text, bytes or mediaDict for an audio or video)
    sequence: List[Union[bytes, str, Image.Image, dict]]


@dataclass_slot
class AVLMEnergonQASample(Sample):
    contexts: List[str]
    answers: Optional[List[str]] = None
    answer_weights: Optional[torch.Tensor] = None

    audios: Optional[List[Union[bytes, dict]]] = None
    videos: Optional[List[Union[bytes, dict]]] = None
    images: Optional[List[Image.Image]] = None


@dataclass
class AVLMSample:
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
    image_attention_mask: Optional[torch.tensor] = None


@dataclass
class AVLMRawBatch:
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
    image_attention_mask: Optional[torch.tensor] = None


@dataclass
class AVLMSampleConfig(MultiModalSampleConfig):
    model_id: str = field(default="llava-hf/llava-v1.6-vicuna-7b-hf")
    audio_token: AudioToken = field(default_factory=AudioToken)
    video_token: VideoToken = field(default_factory=VideoToken)
    """
    For a single video with multiple video and audio streams
    video_audio: video streams tokens are before the audio streams tokens 
    audio_video: audio streams tokens are before the video streams tokens
    interleaved_optimal: space the video tokens and the audio tokens as evenly as possible
    """
    video_audio_token_concatenate_pattern: Literal[
        "video_audio", 
        "audio_video", 
        "interleaved_optimal",] = field(default="video_audio")
    