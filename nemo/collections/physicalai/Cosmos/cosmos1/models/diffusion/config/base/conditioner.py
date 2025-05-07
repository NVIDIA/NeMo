# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0115,C0116,C0301

from typing import Dict, List, Optional

import attrs
import torch

from cosmos1.models.diffusion.conditioner import BaseConditionEntry, TextAttr, VideoConditioner, VideoExtendConditioner
from cosmos1.utils.lazy_config import LazyCall as L
from cosmos1.utils.lazy_config import LazyDict


@attrs.define(slots=False)
class TextConfig:
    obj: LazyDict = L(TextAttr)()  # No arguments
    dropout_rate: float = 0.2
    input_keys: List[str] = attrs.field(factory=lambda: ["t5_text_embeddings", "t5_text_mask"])


class BooleanFlag(BaseConditionEntry):
    def __init__(self, output_key: Optional[str] = None):
        super().__init__()
        self.output_key = output_key

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        del args, kwargs
        key = self.output_key if self.output_key else self.input_key
        return {key: self.flag}

    def random_dropout_input(
        self, in_tensor: torch.Tensor, dropout_rate: Optional[float] = None, key: Optional[str] = None
    ) -> torch.Tensor:
        del key
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        self.flag = torch.bernoulli((1.0 - dropout_rate) * torch.ones(1)).bool().to(device=in_tensor.device)
        return in_tensor


class ReMapkey(BaseConditionEntry):
    def __init__(self, output_key: Optional[str] = None, dtype: Optional[str] = None):
        super().__init__()
        self.output_key = output_key
        self.dtype = {
            None: None,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
            "half": torch.float16,
            "float16": torch.float16,
            "int": torch.int32,
            "long": torch.int64,
        }[dtype]

    def forward(self, element: torch.Tensor) -> Dict[str, torch.Tensor]:
        key = self.output_key if self.output_key else self.input_key
        if isinstance(element, torch.Tensor):
            element = element.to(dtype=self.dtype)
        return {key: element}


class FrameRepeatAttr(BaseConditionEntry):
    def __init__(self):
        super().__init__()

    def forward(self, frame_repeat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "frame_repeat": frame_repeat / 10.0,
        }

    def details(self) -> str:
        return "Frame repeat, Output key: [frame_repeat]"


@attrs.define(slots=False)
class FPSConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `fps`.
    """

    obj: LazyDict = L(ReMapkey)(output_key="fps", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "fps"


@attrs.define(slots=False)
class PaddingMaskConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `padding_mask`.
    """

    obj: LazyDict = L(ReMapkey)(output_key="padding_mask", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "padding_mask"


@attrs.define(slots=False)
class ImageSizeConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `image_size`.
    """

    obj: LazyDict = L(ReMapkey)(output_key="image_size", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "image_size"


@attrs.define(slots=False)
class NumFramesConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `num_frames`.
    """

    obj: LazyDict = L(ReMapkey)(output_key="num_frames", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "num_frames"


@attrs.define(slots=False)
class FrameRepeatConfig:
    """
    Remap and process key from the input dictionary to the output dictionary. For `frame_repeat`.
    """

    obj: LazyDict = L(FrameRepeatAttr)()
    dropout_rate: float = 0.0
    input_key: str = "frame_repeat"


@attrs.define(slots=False)
class VideoCondBoolConfig:
    obj: LazyDict = L(BooleanFlag)(output_key="video_cond_bool")
    dropout_rate: float = 0.2
    input_key: str = "fps"  # This is a placeholder, we never use this value
    # Config below are for long video generation only

    # Sample PPP... from IPPP... sequence
    sample_tokens_start_from_p_or_i: bool = False


@attrs.define(slots=False)
class LatentConditionConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `latent condition`.
    """

    obj: LazyDict = L(ReMapkey)(output_key="latent_condition", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "latent_condition"


@attrs.define(slots=False)
class LatentConditionSigmaConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `latent condition`.
    """

    obj: LazyDict = L(ReMapkey)(output_key="latent_condition_sigma", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "latent_condition_sigma"


BaseVideoConditionerConfig: LazyDict = L(VideoConditioner)(
    text=TextConfig(),
)

VideoConditionerFpsSizePaddingConfig: LazyDict = L(VideoConditioner)(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
)

VideoExtendConditionerConfig: LazyDict = L(VideoExtendConditioner)(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
    video_cond_bool=VideoCondBoolConfig(),
)

VideoConditionerFpsSizePaddingFrameRepeatConfig: LazyDict = L(VideoConditioner)(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
    frame_repeat=FrameRepeatConfig(),
)

VideoExtendConditionerFrameRepeatConfig: LazyDict = L(VideoExtendConditioner)(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
    video_cond_bool=VideoCondBoolConfig(),
    frame_repeat=FrameRepeatConfig(),
)
