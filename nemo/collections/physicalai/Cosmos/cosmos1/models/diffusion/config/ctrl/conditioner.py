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

from cosmos1.models.diffusion.conditioner import VideoConditionerWithCtrl
from cosmos1.models.diffusion.config.base.conditioner import (
    FPSConfig,
    ImageSizeConfig,
    NumFramesConfig,
    PaddingMaskConfig,
    TextConfig,
    VideoCondBoolConfig,
)
from cosmos1.models.diffusion.datasets.augmentors.control_input import (
    AddControlInput,
    AddControlInputCanny,
    AddControlInputDepth,
    AddControlInputHumanKpts,
    AddControlInputIdentity,
    AddControlInputMask,
    AddControlInputSeg,
    AddControlInputUpscale,
)
from cosmos1.utils.lazy_config import LazyCall as L
from cosmos1.utils.lazy_config import LazyDict

CTRL_HINT_KEYS_SINGLE = [
    "control_input_canny",
    "control_input_blur",
    "control_input_depth",
    "control_input_human_kpts",
    "control_input_segmentation",
    "control_input_mask",
    "control_input_upscale",
    "control_input_identity",
]

CTRL_HINT_KEYS_MULTI = [
    "control_input_canny_blur",
    "control_input_depth_segmentation",
]

CTRL_HINT_KEYS = CTRL_HINT_KEYS_SINGLE + CTRL_HINT_KEYS_MULTI

CTRL_HINT_KEYS_COMB = {
    "control_input_canny_blur": [AddControlInputCanny, AddControlInput],
    "control_input_depth_segmentation": [AddControlInputDepth, AddControlInputSeg],
    "control_input_blur": [AddControlInput],
    "control_input_canny": [AddControlInputCanny],
    "control_input_depth": [AddControlInputDepth],
    "control_input_human_kpts": [AddControlInputHumanKpts],
    "control_input_segmentation": [AddControlInputSeg],
    "control_input_mask": [AddControlInputMask],
    "control_input_upscale": [AddControlInputUpscale],
    "control_input_identity": [AddControlInputIdentity],
}


BaseVideoConditionerWithCtrlConfig: LazyDict = L(VideoConditionerWithCtrl)(
    text=TextConfig(),
)

VideoConditionerFpsSizePaddingWithCtrlConfig: LazyDict = L(VideoConditionerWithCtrl)(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
    video_cond_bool=VideoCondBoolConfig(),
)
