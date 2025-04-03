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

from typing import List, Optional

import attrs

from cosmos1.utils.lazy_config import LazyCall as L


@attrs.define
class GaussianBlurConfig:
    """Configuration for Gaussian blur"""

    use_random: bool = False
    # if use_random is False, then optionally define the param values
    ksize: int = 25
    sigmaX: float = 12.5

    # if use_random is True, then optionally define the range
    ksize_min: int = 21
    ksize_max: int = 29
    sigmaX_min: float = 10.5
    sigmaX_max: float = 14.5


LowGaussianBlurConfig = L(GaussianBlurConfig)(ksize=21, sigmaX=10.5)


@attrs.define
class GuidedFilterConfig:
    """Configuration for Guided filter"""

    use_random: bool = False
    # if use_random is False, then optionally define the param values
    radius: int = 45
    eps: float = 0.15
    scale: int = 10

    # if use_random is True, then optionally define the range
    radius_min: int = 41
    radius_max: int = 49
    eps_min: float = 0.1
    eps_max: float = 0.2
    scale_min: int = 3
    scale_max: int = 18


@attrs.define
class BilateralFilterConfig:
    """Configuration for Bilateral filter"""

    use_random: bool = False
    # if use_random is False, then optionally define the param values
    d: int = 30
    sigma_color: int = 150
    sigma_space: int = 100
    iter: int = 1

    # if use_random is True, then optionally define the range
    d_min: int = 15
    d_max: int = 50
    sigma_color_min: int = 100
    sigma_color_max: int = 300
    sigma_space_min: int = 50
    sigma_space_max: int = 150
    iter_min: int = 1
    iter_max: int = 4


@attrs.define
class MedianBlurConfig:
    """Configuration for Median blur"""

    use_random: bool = False
    # if use_random is False, then optionally define the param values
    ksize: int = 11

    # if use_random is True, then optionally define the range
    ksize_min: int = 9
    ksize_max: int = 15


@attrs.define
class LaplacianOfGaussianConfig:
    """Configuration for LoG filter"""

    use_random: bool = False
    # if use_random is False, then optionally define the param values
    ksize: int = 5
    sigma: float = 1.4
    binarize: bool = False
    threshold: float = 0.0

    # if use_random is True, then optionally define the range
    ksize_min: int = 3
    ksize_max: int = 7
    sigma_min: float = 0.5
    sigma_max: float = 3.0
    threshold_min: float = 10.0
    threshold_max: float = 30.0


@attrs.define
class AnisotropicDiffusionConfig:
    """Configuration for Anisotropic Diffusion"""

    use_random: bool = False
    alpha: float = 0.25
    K: float = 0.15
    niters: int = 12

    # if use_random is True, then optionally define the range
    alpha_min: float = 0.2
    alpha_max: float = 0.3
    K_min: float = 0.1
    K_max: float = 0.2
    niters_min: int = 10
    niters_max: int = 14


@attrs.define
class BlurCombinationConfig:
    """Configuration for a combination of blurs with associated probability"""

    # list of choices are:  ["gaussian", "guided", "bilateral", "median", "log", "anisotropic"]
    # the corresponding config must be defined for each item in this blur_types list
    blur_types: List[str]
    probability: float
    gaussian_blur: Optional[GaussianBlurConfig] = None
    guided_filter: Optional[GuidedFilterConfig] = None
    bilateral_filter: Optional[BilateralFilterConfig] = None
    median_blur: Optional[MedianBlurConfig] = None
    log: Optional[LaplacianOfGaussianConfig] = None
    anisotropic_diffusion: Optional[AnisotropicDiffusionConfig] = None


@attrs.define
class BlurAugmentorConfig:
    """Configuration for blur augmentation with multiple combinations"""

    # probabilities from the list of combinations should add up to 1.0
    blur_combinations: List[BlurCombinationConfig] = []
    downscale_factor: List[int] = [1]
