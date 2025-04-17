# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# pylint: disable=C0115,C0116,C0301

import attrs
from nemo.collections.physicalai.tokenizer.losses.loss import ReduceMode


@attrs.define(slots=False)
class ColorConfig:
    # Color (RGB) basic loss and its weight schedule.
    norm: str = "L1"
    boundaries: list[int] = [0]
    values: list[float] = [1.0]  # TODO: confirm with fitsum


@attrs.define(slots=False)
class PerceptualConfig:
    lpips_boundaries: list[int] = [500000]  # TODO: confirm with fitsum
    lpips_values: list[float] = [0.1, 0.073]
    # Layer weights for linearly combining the multi-layer vgg-based losses.
    layer_weights: list[float] = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5]
    # Gram loss, whether to turn on, and what weights to use.
    gram_loss: bool = True
    gram_boundaries: list[int] = [500000]
    gram_values: list[float] = [0.0, 0.062]
    # Corr loss, whether to turn on, and what weights to use.
    corr_loss: bool = False
    corr_boundaries: list[int] = [0]
    corr_values: list[float] = [0.0]
    # In the example training memory usage dropped from 64.03 GiB to 60.54 GiB
    # with checkpointing enabled for this loss for about 3.2% slowdown.
    # With checkpointing this and PerceptualLoss memory usage dropped
    # from 64.03 GiB to 52.94 GiB for about 18% slowdown
    # more details in MR:949
    checkpoint_activations: bool = False


@attrs.define(slots=False)
class FlowConfig:
    # Flow loss and its weight schedule.
    boundaries: list[int] = [250000]  # TODO: confirm with fitsum
    values: list[float] = [0.0, 0.01]
    scale: int = 2
    # Flow loss depends on RAFT, as such it requires a specific dtype.
    dtype: str = "bfloat16"
    # In the example training memory usage dropped from 28GB to 23GB
    # with checkpointing enabled for this loss
    # With checkpointing this and PerceptualLoss memory usage dropped
    # from 64.03 GiB to 52.94 GiB for about 18% slowdown
    # more details in MR:949
    checkpoint_activations: bool = False


@attrs.define(slots=False)
class VideoLoss:
    # The combined loss function, and its reduction mode.
    color: ColorConfig = ColorConfig()
    perceptual: PerceptualConfig = PerceptualConfig()
    flow: FlowConfig = FlowConfig()
    reduce: str = ReduceMode.MEAN.value  # model.config.loss.config.reduce={'MEAN', 'SUM', 'SUM_PER_FRAME'}
    start_human_mask: int = 500000  # iteration when we enable human loss mask: TODO:
