# -*- coding: utf-8 -*-
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

from os.path import expanduser
from typing import Optional

from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor

from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import (
    AxisKind,
    AxisType,
    ClassificationTarget,
    Index,
    NeuralType,
    NormalizedImageValue,
    StringLabel,
)
from nemo.utils.decorators import add_port_docs

__all__ = ['MNISTDataLayer']


class MNISTDataLayer(DataLayerNM, Dataset):
    """
    A "thin DataLayer" -  wrapper around the torchvision's MNIST dataset.
    """

    def __init__(
        self,
        height: int = 28,
        width: int = 28,
        data_folder: str = "~/data/mnist",
        train: bool = True,
        name: Optional[str] = None,
        batch_size: int = 64,
        shuffle: bool = True,
    ):
        """
        Initializes the MNIST datalayer.

        Args:
            height: image height (DEFAULT: 28)
            width: image width (DEFAULT: 28)
            data_folder: path to the folder with data, can be relative to user (DEFAULT: "~/data/mnist")
            train: use train or test splits (DEFAULT: True)
            name: Name of the module (DEFAULT: None)
            batch_size: size of batch (DEFAULT: 64) [PARAMETER OF DATALOADER]
            shuffle: shuffle data (DEFAULT: True) [PARAMETER OF DATALOADER]
        """
        # Call the base class constructor of DataLayer.
        DataLayerNM.__init__(self, name=name)

        # Store height and width.
        self._height = height
        self._width = width

        # Create transformations: up-scale and transform to tensors.
        mnist_transforms = Compose([Resize((self._height, self._width)), ToTensor()])

        # Get absolute path.
        abs_data_folder = expanduser(data_folder)

        # Create the MNIST dataset object.
        self._dataset = MNIST(root=abs_data_folder, train=train, download=True, transform=mnist_transforms)

        # Remember the params passed to DataLoader. :]
        self._batch_size = batch_size
        self._shuffle = shuffle

        # Class names.
        labels = 'Zero One Two Three Four Five Six Seven Eight Nine'.split(' ')
        word_to_ix = {labels[i]: i for i in range(10)}

        # Reverse mapping.
        self._ix_to_word = {value: key for (key, value) in word_to_ix.items()}

    @property
    @add_port_docs()
    def output_ports(self):
        """
        Creates definitions of output ports.
        By default, it sets image width and height to 32.
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
            "targets": NeuralType(tuple('B'), elements_type=ClassificationTarget()),  # Target are ints!
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

    @property
    def dataset(self):
        """
        Returns:
            Self - just to be "compatible" with the current NeMo train action.
        """
        return self  # ! Important - as we want to use this __getitem__ method!
