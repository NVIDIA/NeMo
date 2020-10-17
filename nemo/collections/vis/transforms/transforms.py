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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

import torch
from hydra.core.config_store import ConfigStore
from hydra.types import ObjectConf

# Create the config store instance.
cs = ConfigStore.instance()


class Transform(ABC):
    """ Abstract class defining the transform interface. """

    @abstractmethod
    def __call__(self, batch: Any) -> Any:
        pass


class Compose:
    """ Class for composing transforms. """

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, batch: Any) -> Any:
        """ Processes the input batch by transforms - one by one. """
        for t in self.transforms:
            batch = t(batch)
        return batch


class ToTensor(Transform):
    """ Transform responsible for tokenization. """

    def __call__(self, batch: List[int]) -> torch.Tensor:
        # Change to torch tensor.
        return torch.tensor(batch)


@dataclass
class ToTensorConfig:
    # Target class name.
    _target_: str = "nemo.collections.vis.transforms.ToTensor"


# Register the config.
cs.store(
    group="nemo.collections.vis.transforms",
    name="ToTensor",
    node=ObjectConf(target="nemo.collections.vis.transforms.ToTensor", params=ToTensorConfig()),
)
