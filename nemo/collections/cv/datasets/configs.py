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

from dataclasses import dataclass

from hydra.types import ObjectConf
from hydra.core.config_store import ConfigStore

# Create the config store instance.
cs = ConfigStore.instance()


@dataclass
class MNISTConfig:
    """
    Structured config for MNIST dataset.

    For more details please refer to:
    https://pytorch.org/docs/stable/torchvision/datasets.html#mnist
    http://yann.lecun.com/exdb/mnist/

    Args:
        _target_: specification of TorchVision dataset class
        root: path to the folder with data, can be relative to user (DEFAULT: "~/data/mnist")
        train: use train or test splits (DEFAULT: True)
        download: download the data (DEFAULT: True)
    """

    # Dataset target class name.
    _target_: str = "torchvision.datasets.MNIST"
    # Original MNIST params
    root: str = "~/data/mnist"
    train: bool = True
    download: bool = True


# Register the config.
cs.store(
    group="nemo.collections.cv.datasets",
    name="MNIST",
    node=ObjectConf(target="torchvision.datasets.MNIST", params=MNISTConfig()),
)
