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
from typing import Callable, Optional


@dataclass
class MultiModalToken:
    """
    Base class for multimodal tokens representing different media types.
    """

    token_str: str
    token_index: int
    media_type: str
    use_start_end: bool
    encoder_fn: Optional[Callable] = None


@dataclass
class Qwen2VLImageToken(MultiModalToken):
    """Image Token class"""

    token_str: str = "<|image_pad|>"
    token_index: int = -200
    media_type: str = "image"
    use_start_end: bool = False


@dataclass
class Qwen2VLVideoToken(MultiModalToken):
    """Video Token class"""

    token_str: str = "<|video_pad|>"
    token_index: int = -300
    media_type: str = "video"
    use_start_end: bool = False


@dataclass
class Qwen2VLEndOfTextToken(MultiModalToken):
    """End-Of-Text Token class"""

    token_str: str = "<|endoftext|>"
    token_index: int = 151643
    media_type: str = None
    use_start_end: bool = False
    lstrip: bool = False
    normalized: bool = False
    rstrip: bool = False
    single_word: bool = False


@dataclass
class Qwen2VLObjectRefStartToken(MultiModalToken):
    """Object-Ref-Start Token class"""

    token_str: str = "<|object_ref_start|>"
    token_index: int = 151646
    media_type: str = None
    use_start_end: bool = False
    lstrip: bool = False
    normalized: bool = False
    rstrip: bool = False
    single_word: bool = False


@dataclass
class Qwen2VLObjectRefEndToken(MultiModalToken):
    """Object-Ref-End Token class"""

    token_str: str = "<|object_ref_end|>"
    token_index: int = 151647
    media_type: str = None
    use_start_end: bool = False
    lstrip: bool = False
    normalized: bool = False
    rstrip: bool = False
    single_word: bool = False


@dataclass
class Qwen2VLBoxStartToken(MultiModalToken):
    """Box-Start Token class"""

    token_str: str = "<|box_start|>"
    token_index: int = 151648
    media_type: str = None
    use_start_end: bool = False
    lstrip: bool = False
    normalized: bool = False
    rstrip: bool = False
    single_word: bool = False


@dataclass
class Qwen2VLBoxEndToken(MultiModalToken):
    """Box-End Token class"""

    token_str: str = "<|box_end|>"
    token_index: int = 151649
    media_type: str = None
    use_start_end: bool = False
    lstrip: bool = False
    normalized: bool = False
    rstrip: bool = False
    single_word: bool = False


@dataclass
class Qwen2VLQuadStartToken(MultiModalToken):
    """Quad-Start Token class"""

    token_str: str = "<|quad_start|>"
    token_index: int = 151650
    media_type: str = None
    use_start_end: bool = False
    lstrip: bool = False
    normalized: bool = False
    rstrip: bool = False
    single_word: bool = False


@dataclass
class Qwen2VLQuadEndToken(MultiModalToken):
    """Quad-End Token class"""

    token_str: str = "<|quad_end|>"
    token_index: int = 151651
    media_type: str = None
    use_start_end: bool = False
    lstrip: bool = False
    normalized: bool = False
    rstrip: bool = False
    single_word: bool = False


@dataclass
class Qwen2VLVisionStartToken(MultiModalToken):
    """Vision-Start Token class"""

    token_str: str = "<|vision_start|>"
    token_index: int = 151652
    media_type: str = None
    use_start_end: bool = False
    lstrip: bool = False
    normalized: bool = False
    rstrip: bool = False
    single_word: bool = False


@dataclass
class Qwen2VLVisionEndToken(MultiModalToken):
    """Vision-End Token class"""

    token_str: str = "<|vision_end|>"
    token_index: int = 151653
    media_type: str = None
    use_start_end: bool = False
    lstrip: bool = False
    normalized: bool = False
    rstrip: bool = False
    single_word: bool = False


# Constants for token indexing and special token mapping
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = Qwen2VLImageToken.token_index
VIDEO_TOKEN_INDEX = Qwen2VLVideoToken.token_index
OBJECT_REF_START_TOKEN_INDEX = Qwen2VLObjectRefStartToken.token_index
OBJECT_REF_END_TOKEN_INDEX = Qwen2VLObjectRefEndToken.token_index
BOX_START_TOKEN_INDEX = Qwen2VLBoxStartToken.token_index
BOX_END_TOKEN_INDEX = Qwen2VLBoxEndToken.token_index
QUAD_START_TOKEN_INDEX = Qwen2VLQuadStartToken.token_index
QUAD_END_TOKEN_INDEX = Qwen2VLQuadEndToken.token_index
VISION_START_TOKEN_INDEX = Qwen2VLVisionStartToken.token_index
VISION_END_TOKEN_INDEX = Qwen2VLVisionEndToken.token_index

SPECIAL_TOKEN_MAP = [
    (Qwen2VLImageToken.token_str, Qwen2VLImageToken.token_index),
    (Qwen2VLVideoToken.token_str, Qwen2VLVideoToken.token_index),
    (Qwen2VLObjectRefStartToken.token_str, Qwen2VLObjectRefStartToken.token_index),
    (Qwen2VLObjectRefEndToken.token_str, Qwen2VLObjectRefEndToken.token_index),
    (Qwen2VLBoxStartToken.token_str, Qwen2VLBoxStartToken.token_index),
    (Qwen2VLBoxEndToken.token_str, Qwen2VLBoxEndToken.token_index),
    (Qwen2VLQuadStartToken.token_str, Qwen2VLQuadStartToken.token_index),
    (Qwen2VLQuadEndToken.token_str, Qwen2VLQuadEndToken.token_index),
    (Qwen2VLVisionStartToken.token_str, Qwen2VLVisionStartToken.token_index),
    (Qwen2VLVisionEndToken.token_str, Qwen2VLVisionEndToken.token_index),
]
