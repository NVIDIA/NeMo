# -*- coding: utf-8 -*-
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

from os.path import expanduser
from typing import Optional

from omegaconf import OmegaConf, MISSING
from hydra.utils import instantiate

from torchvision.transforms import Compose, Resize, ToTensor

from nemo.core.classes import Dataset
from nemo.core.neural_types.axes import AxisKind, AxisType
from nemo.core.neural_types.elements import ClassificationTarget, Index, NormalizedImageValue, StringLabel
from nemo.core.neural_types.neural_type import NeuralType


@dataclass
class xMNISTDatasetConfig:
    """
    The "default" structured config for xMNISTDataset class.

    Args:
        _target_: specification of TorchVision dataset class (DEFAULT: MISSING)
        height: image height (DEFAULT: 28)
        width: image width (DEFAULT: 28)
        data_folder: path to the folder with data, can be relative to user (DEFAULT: MISSING)
        train: use train or test splits (DEFAULT: True)
        download: download the data (DEFAULT: True)
        labels: Labels of the classes (DEFAULT: MISSING)
    """
    _target_: str = MISSING

class xMNISTDataset(Dataset):
    """
    A generic "thin wrapper" around the torchvision's various variations of the MNIST dataset.
    
    Please analyse the available (x)MNISTDatasetConfig classes.
    """

    def __init__(self, cfg: xMNISTDatasetConfig = xMNISTDatasetConfig()):
        """
        Initializes the MNIST dataset.

        Args:
            cfg: Configuration object of type MNISTDatasetConfig.

        """
        # Call the base class constructor of Dataset.
        Dataset.__init__(self)  # , name=name)

        # Store height and width.
        self._height = cfg.height
        self._width = cfg.width

        # Create transformations: scale and transform to tensors.
        transforms = Compose([Resize((self._height, self._width)), ToTensor()])

        # Get absolute path.
        abs_data_folder = expanduser(cfg.data_folder)

        # Create the config object for a given (x)MNIST.
        xmnist_config = OmegaConf.create({
            "_target_": cfg._target_,
            "root": abs_data_folder,
            "train": cfg.train,
            "download": cfg.download
            })

        # Create the (x)MNIST dataset object.
        self._dataset = instantiate(xmnist_config, transform=transforms)

        # Create mapping from class id to name.
        labels = cfg.labels.split(",")
        self._ix_to_word = {i: l for i, l in zip(range(len(labels)), labels)}

    @property
    def output_types(self):
        """
        Creates definitions of output ports.
        """
        return {
            "indices": NeuralType(tuple('B'), elements_type=Index()),
            "images": NeuralType(
                axes=(
                    AxisType(kind=AxisKind.Batch),
                    AxisType(kind=AxisKind.Channel, size=1),
                    AxisType(kind=AxisKind.Height, size=self._height),
                    AxisType(kind=AxisKind.Width, size=self._width),
                ),
                elements_type=NormalizedImageValue(),  # float, <0-1>
            ),
            "targets": NeuralType(tuple('B'), elements_type=ClassificationTarget()),  # Targets are ints!
            "labels": NeuralType(tuple('B'), elements_type=StringLabel()),  # Labels is string!
        }

    def __len__(self):
        """
        Returns:
            Length of the dataset.
        """
        return len(self._dataset)

    def __getitem__(self, index: int):
        """
        Returns a single sample.

        Args:
            index: index of the sample to return.
        """
        # Get image and target.
        img, target = self._dataset.__getitem__(index)

        # Return sample.
        return index, img, target, self._ix_to_word[target]

    @property
    def ix_to_word(self):
        """
        Returns:
            Dictionary with mapping of target indices (int) to labels (class names as strings)
            that can we used by other modules.
        """
        return self._ix_to_word
