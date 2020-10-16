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

from dataclasses import MISSING, dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from hydra.types import ObjectConf

# Create the config store instance.
cs = ConfigStore.instance()


@dataclass
class ResizeConfig:
    interpolation: int = 2
    # size: List[int] = MISSING
    # Target class name.
    _target_: str = "torchvision.transforms.Resize"


# Register the config.
cs.store(
    group="nemo.collections.cv.transforms",
    name="Resize",
    node=ObjectConf(target="torchvision.transforms.Resize", params=ResizeConfig()),
)


@dataclass
class ToTensorConfig:
    # Target class name.
    _target_: str = "torchvision.transforms.ToTensor"


# Register the config.
cs.store(
    group="nemo.collections.cv.transforms",
    name="ToTensor",
    node=ObjectConf(target="torchvision.transforms.ToTensor", params=ToTensorConfig()),
)


@dataclass
class NormalizeConfig:
    inplace: bool = False
    # mean: List[int] = MISSING
    # std: List[int] = MISSING
    # Target class name.
    _target_: str = "torchvision.transforms.Normalize"


# Register the config.
cs.store(
    group="nemo.collections.cv.transforms",
    name="ToTensor",
    node=ObjectConf(target="torchvision.transforms.Normalize", params=NormalizeConfig()),
)
