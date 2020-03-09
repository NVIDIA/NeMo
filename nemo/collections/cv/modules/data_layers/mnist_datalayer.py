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

import torch
from torchvision import datasets, transforms

from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import AxisKind, AxisType, LabelsType, NeuralType, NormalizedValueType
from nemo.utils.decorators import add_port_docs

__all__ = ['MNISTDataLayer']


class MNISTDataLayer(DataLayerNM):
    """Wrapper around torchvision's MNIST dataset.
    """

    def __init__(self, batch_size, data_folder="~/data/mnist", train=True, shuffle=True):
        """
        Initializes the MNIST datalayer.

        Args:
            batch_size: size of batch
            data_folder: path to the folder with data, can be relative to user.
            train: use train or test splits
            shuffle: shuffle data (True by default)
        """
        # Call the base class constructor.
        super().__init__()

        self._batch_size = batch_size
        self._train = train
        self._shuffle = shuffle

        # Get absolute path.
        abs_data_folder = expanduser(data_folder)

        # Up-scale and transform to tensors.
        self._transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

        self._dataset = datasets.MNIST(
            root=abs_data_folder, train=self._train, download=True, transform=self._transforms
        )

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
                    AxisType(kind=AxisKind.Channel, size=1),
                    AxisType(kind=AxisKind.Height, size=32),
                    AxisType(kind=AxisKind.Width, size=32),
                ),
                elements_type=NormalizedValueType(),
            ),
            "targets": NeuralType(tuple('B'), elements_type=LabelsType()),
        }

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None
