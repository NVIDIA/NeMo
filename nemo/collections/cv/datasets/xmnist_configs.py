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
    Structured config for MNIST dataset.

    For more details please refer to:
    https://pytorch.org/docs/stable/torchvision/datasets.html#mnist
    http://yann.lecun.com/exdb/mnist/

    Args:
        _target_: specification of TorchVision dataset class
        height: image height (DEFAULT: 28)
        width: image width (DEFAULT: 28)
        data_folder: path to the folder with data, can be relative to user (DEFAULT: "~/data/mnist")
        train: use train or test splits (DEFAULT: True)
        download: download the data (DEFAULT: True)
        labels: Labels of the classes (coma separated).
    """
    _target_: str = "torchvision.datasets.MNIST"
    height: int = 28
    width: int = 28
    data_folder: str = "~/data/mnist"
    train: bool = True
    download: bool = True
    labels: str = "Zero,One,Two,Three,Four,Five,Six,Seven,Eight,Nine"

# Register the MNIST config.
cs.store(
    group="nemo.collections.cv.datasets",
    name="MNIST",
    node=ObjectConf(target="nemo.collections.cv.datasets.xMNISTDataset", params=MNISTDatasetConfig()),
)


@dataclass
class FashionMNISTDatasetConfig:
    """
    Structured config for Fashion-MNIST dataset - a variation of MNIST provided by Zelando.

    For more details please refer to:
    https://pytorch.org/docs/stable/torchvision/datasets.html#fashion-mnist
    https://github.com/zalandoresearch/fashion-mnist

    Args:
        _target_: specification of TorchVision dataset class
        height: image height (DEFAULT: 28)
        width: image width (DEFAULT: 28)
        data_folder: path to the folder with data, can be relative to user (DEFAULT: "~/data/fashion-mnist")
        train: use train or test splits (DEFAULT: True)
        download: download the data (DEFAULT: True)
        labels: Labels of the classes (coma separated).
    """
    _target_: str = "torchvision.datasets.FashionMNIST"
    height: int = 28
    width: int = 28
    data_folder: str = "~/data/fashion-mnist"
    train: bool = True
    download: bool = True
    labels: str = "T-shirt/top,Trouser,Pullover,Dress,Coat,Sandal,Shirt,Sneaker,Bag,Ankle boot"


# Register the FashionMNIST config.
cs.store(
    group="nemo.collections.cv.datasets",
    name="FashionMNIST",
    node=ObjectConf(target="nemo.collections.cv.datasets.xMNISTDataset", params=FashionMNISTDatasetConfig()),
)


@dataclass
class KMNISTDatasetConfig:
    """
    Structured config for Kuzushiji-MNIST dataset - a variation of MNIST with 10 Hiragana characters.
    (28x28 grayscale, 70,000 images)

    For more details please refer to:
    https://pytorch.org/docs/stable/torchvision/datasets.html#kmnist
    https://github.com/rois-codh/kmnist

    Args:
        _target_: specification of TorchVision dataset class
        height: image height (DEFAULT: 28)
        width: image width (DEFAULT: 28)
        data_folder: path to the folder with data, can be relative to user (DEFAULT: "~/data/fashion-mnist")
        train: use train or test splits (DEFAULT: True)
        download: download the data (DEFAULT: True)
        labels: Labels of the classes (coma separated).
    """
    _target_: str = "torchvision.datasets.KMNIST"
    height: int = 28
    width: int = 28
    data_folder: str = "~/data/kuzushiji-mnist"
    train: bool = True
    download: bool = True
    labels: str = "0,1,2,3,4,5,6,7,8,9"


# Register the KMNIST config.
cs.store(
    group="nemo.collections.cv.datasets",
    name="KMNIST",
    node=ObjectConf(target="nemo.collections.cv.datasets.xMNISTDataset", params=KMNISTDatasetConfig()),
)
