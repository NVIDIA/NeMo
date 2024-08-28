from dataclasses import dataclass
from typing import Callable, Optional

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
VIDEO_TOKEN_INDEX = -300


@dataclass
class MultiModalToken:
    token_str: str
    token_index: int
    media_type: str
    use_start_end: bool
    encoder_fn: Optional[Callable] = None


@dataclass
class ImageToken(MultiModalToken):
    token_str: str = "<image>"
    token_index: int = -200
    media_type: str = "image"
    use_start_end: bool = False


@dataclass
class VideoToken(MultiModalToken):
    token_str: str = "<video>"
    token_index: int = -300
    media_type: str = "video"
    use_start_end: bool = False
