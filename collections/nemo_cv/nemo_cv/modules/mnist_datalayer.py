# Copyright (C) tkornuta, NVIDIA AI Applications Team. All Rights Reserved.
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

import torch

from torch.utils.data import Dataset
from torchvision import transforms, datasets

from nemo.backends.pytorch.nm import TrainableNM, NonTrainableNM, LossNM,\
    DataLayerNM

from nemo.core import NeuralType, BatchTag, ChannelTag, HeightTag, WidthTag,\
    AxisType, DeviceType, LogProbabilityTag


class MNISTDataLayer(DataLayerNM):
    """Wrapper around torchvision's MNIST dataset.

    Args:
        batch_size (int)
        root (str): Where to store the dataset
        train (bool)
        shuffle (bool)
    """

    @staticmethod
    def create_ports(input_size=(32, 32)):
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
            self, *,
            batch_size,
            root,
            train=True,
            shuffle=True,
            **kwargs
    ):
        # Passing the default params to base init call.
        DataLayerNM.__init__(self, **kwargs)

        self._batch_size = batch_size
        self._train = train
        self._shuffle = shuffle
        self._root = root

        # Up-scale and transform to tensors.
        self._transforms = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()])

        self._dataset = datasets.MNIST(root=self._root, train=self._train,
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
