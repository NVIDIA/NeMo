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

from cosmos1.models.diffusion.config.ctrl.blurs import (
    BilateralFilterConfig,
    BlurAugmentorConfig,
    BlurCombinationConfig,
)
from cosmos1.utils.lazy_config import LazyCall as L

# predefined BilateralFilterConfig with different strength level
NoFilterConfig = L(BilateralFilterConfig)(use_random=False, d=1, sigma_color=1, sigma_space=1, iter=1)

LowBilateralFilterConfig = L(BilateralFilterConfig)(use_random=False, d=15, sigma_color=100, sigma_space=50, iter=1)

MediumBilateralFilterConfig = L(BilateralFilterConfig)(
    use_random=False, d=30, sigma_color=150, sigma_space=100, iter=1
)

HighBilateralFilterConfig = L(BilateralFilterConfig)(use_random=False, d=50, sigma_color=300, sigma_space=150, iter=1)

BilateralOnlyBlurAugmentorConfig = {}
for strength, blur_config in zip(
    ["none", "very_low", "low", "medium", "high", "very_high"],
    [
        NoFilterConfig,
        LowBilateralFilterConfig,
        LowBilateralFilterConfig,
        MediumBilateralFilterConfig,
        HighBilateralFilterConfig,
        HighBilateralFilterConfig,
    ],
):
    BlurConfig = L(BlurCombinationConfig)(
        blur_types=["bilateral"],
        probability=1.0,
        bilateral_filter=blur_config,
    )
    downscale_factor = {
        "none": 1,
        "very_low": 1,
        "low": 4,
        "medium": 2,
        "high": 1,
        "very_high": 4,
    }
    BilateralOnlyBlurAugmentorConfig[strength] = L(BlurAugmentorConfig)(
        blur_combinations=[BlurConfig],
        downscale_factor=[downscale_factor[strength]],
    )
