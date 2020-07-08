# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

from typing import Tuple

from dataclasses import dataclass

__all__ = ['AdamConfig']

@dataclass
class AdamConfig:
    """
    Default configuration for Adam optimizer.
    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).

    ..note:
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html?highlight=adam#torch.optim.Adam
    """
    lr: float=0.001
    betas: Tuple[float, float]=(0.9, 0.999)
    eps: float=1e-08
    weight_decay: float=0
    amsgrad: bool=False