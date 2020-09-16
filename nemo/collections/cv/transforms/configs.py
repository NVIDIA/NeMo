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

from typing import Optional, Any, List
from dataclasses import dataclass, field, MISSING

from hydra.types import ObjectConf
from hydra.core.config_store import ConfigStore

# Create the config store instance.
cs = ConfigStore.instance()


@dataclass
class ResizeConfig:
    # Target class name.
    _target_: str = "torchvision.transforms.Resize"
    interpolation: int = 2
    # size: List[int] = MISSING
    # size: Any = MISSING


# TypeError: non-default argument 'size' follows default argument

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
    # Target class name.
    _target_: str = "torchvision.transforms.Normalize"
    inplace: bool = False
    # mean: List[int] = MISSING
    # std: List[int] = MISSING


# Register the config.
cs.store(
    group="nemo.collections.cv.transforms",
    name="ToTensor",
    node=ObjectConf(target="torchvision.transforms.Normalize", params=NormalizeConfig()),
)
