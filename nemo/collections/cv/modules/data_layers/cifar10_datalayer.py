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

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor

from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import AxisKind, AxisType, NeuralType, NormalizedImageType, ClassificationTargetType
from nemo.utils.decorators import add_port_docs

__all__ = ['CIFAR10DataLayer']


class CIFAR10DataLayer(DataLayerNM, CIFAR10):
    """
    A "thin DataLayer" -  wrapper around the torchvision's CIFAR10 dataset.
    """

    def __init__(
        self, name=None, height=32, width=32, data_folder="~/data/cifar10", train=True, batch_size=64, shuffle=True
    ):
        """
        Initializes the CIFAR10 datalayer.

        Args:
            name: Name of the module (DEFAULT: None)
            height: image height (DEFAULT: 32)
            width: image width (DEFAULT: 32)
            data_folder: path to the folder with data, can be relative to user (DEFAULT: "~/data/cifar10")
            train: use train or test splits (DEFAULT: True)
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

        # Call the base class constructor of MNIST dataset.
        CIFAR10.__init__(self, root=abs_data_folder, train=train, download=True, transform=mnist_transforms)

        # Remember the params passed to DataLoader. :]
        self._batch_size = batch_size
        self._shuffle = shuffle

    @property
    @add_port_docs()
    def output_ports(self):
        """
        Creates definitions of output ports.
        By default, it sets image width and height to 32.
        """
        return {
            "images": NeuralType(
                axes=(
                    AxisType(kind=AxisKind.Batch),
                    AxisType(kind=AxisKind.Channel, size=3),
                    AxisType(kind=AxisKind.Height, size=self._height),
                    AxisType(kind=AxisKind.Width, size=self._width),
                ),
                elements_type=NormalizedImageType(),
            ),
            "targets": NeuralType(tuple('B'), elements_type=ClassificationTargetType()),
        }

    def __len__(self):
        """
        Returns:
            len(Data) - to overwrite the abstract method (which is already overwritten by the other dependency)
        """
        return len(self.data)

    @property
    def dataset(self):
        """
        Returns:
            Self - just to be "compatible" with the current NeMo train action.
        """
        return self
