# Copyright (C) NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Tomasz Kornuta"

from os.path import expanduser
import torch

from torchvision import transforms, datasets

from nemo.backends.pytorch.nm import DataLayerNM

from nemo.core import NeuralType, AxisType, BatchTag, ChannelTag, HeightTag, \
    WidthTag


class MNISTDataLayer(DataLayerNM):
    """Wrapper around torchvision's MNIST dataset.
    """

    @staticmethod
    def create_ports(input_size=(32, 32)):
        """
        Creates definitions of input and output ports.
        By default, it sets HeightTag WidthTag size to 32.
        (TO BE REFACTORED).
        """
        input_ports = {}
        output_ports = {
            "images": NeuralType({0: AxisType(BatchTag),
                                  1: AxisType(ChannelTag, 1),
                                  2: AxisType(HeightTag, input_size[1]),
                                  3: AxisType(WidthTag, input_size[0])}),
            "targets": NeuralType({0: AxisType(BatchTag)})
        }
        return input_ports, output_ports

    def __init__(
        self,
        batch_size,
        data_folder="~/data/mnist",
        train=True,
        shuffle=True
    ):
        """
        Initializes the MNIST datalayer.

        Args:
            batch_size: size of batch
            data_folder: path to the folder with data, can be relative to user.
            train: use train or test splits
            shuffle: shuffle data (True by default)
        """
        # Passing the default params to base init call.
        DataLayerNM.__init__(self)

        self._batch_size = batch_size
        self._train = train
        self._shuffle = shuffle

        # Get absolute path.
        abs_data_folder = expanduser(data_folder)

        # Up-scale and transform to tensors.
        self._transforms = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()])

        self._dataset = datasets.MNIST(root=abs_data_folder, train=self._train,
                                       download=True,
                                       transform=self._transforms)

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None
