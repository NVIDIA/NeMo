# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.

from typing import Any, List

import attrs

from nemo.collections.diffusion.sampler.conditioner import ReMapkey, TextAttr, VideoConditioner


@attrs.define(slots=False)
class TextConfig:

    obj: Any = TextAttr()  # No arguments
    dropout_rate: float = 0.2
    input_keys: List[str] = attrs.field(factory=lambda: ["t5_text_embeddings", "t5_text_mask"])

@attrs.define(slots=False)
class FPSConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `fps`.
    """

    obj: Any = ReMapkey(output_key="fps", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "fps"


@attrs.define(slots=False)
class PaddingMaskConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `fps`.
    """

    obj: Any = ReMapkey(output_key="padding_mask", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "padding_mask"


@attrs.define(slots=False)
class ImageSizeConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `fps`.
    """

    obj: Any = ReMapkey(output_key="image_size", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "image_size"


@attrs.define(slots=False)
class NumFramesConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `num_frames`.
    """

    obj: Any = ReMapkey(output_key="num_frames", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "num_frames"


BaseVideoConditionerConfig: Any = VideoConditioner(
    text=TextConfig(),
)

VideoConditionerFPSConfig: Any = VideoConditioner(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
)

VideoConditionerFpsSizePaddingConfig: Any = VideoConditioner(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
)
