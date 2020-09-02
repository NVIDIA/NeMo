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
class MNISTDatasetConfig:
    """
    Structured config for MNISTDataset class.

    Args:
        height: image height (DEFAULT: 28)
        width: image width (DEFAULT: 28)
        data_folder: path to the folder with data, can be relative to user (DEFAULT: "~/data/mnist")
        train: use train or test splits (DEFAULT: True)
        labels: Labels of the classes.
    """
    _target_: str = "torchvision.datasets.MNIST"
    height: int = 28
    width: int = 28
    data_folder: str = "~/data/mnist"
    train: bool = True
    download: bool = True
    labels: str = "Zero One Two Three Four Five Six Seven Eight Nine"

# Register the MNIST config.
cs.store(
    group="nemo.collections.cv.datasets",
    name="MNIST",
    node=ObjectConf(target="nemo.collections.cv.datasets.xMNISTDataset", params=MNISTDatasetConfig()),
)
