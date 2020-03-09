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

from nemo.backends.pytorch.nm import TrainableNM, NonTrainableNM, LossNM,\
    DataLayerNM

from nemo.core import NeuralType, BatchTag, ChannelTag, HeightTag, WidthTag,\
    AxisType, DeviceType, LogProbabilityTag


class LeNet5(TrainableNM):
    """Classical LeNet-5 model for MNIST image classification.
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "images": NeuralType({0: AxisType(BatchTag),
                                  1: AxisType(ChannelTag, 1),
                                  2: AxisType(HeightTag, 32),
                                  3: AxisType(WidthTag, 32)
                                  })
        }
        output_ports = {
            "predictions": NeuralType({0: AxisType(BatchTag),
                                       1: AxisType(LogProbabilityTag)
                                       })
        }
        return input_ports, output_ports

    def __init__(self):
        """
        Creates the LeNet-5 model.
        """
        super().__init__()

        # Create the LeNet-5 model.
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=(5, 5)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            torch.nn.Conv2d(6, 16, kernel_size=(5, 5)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            torch.nn.Conv2d(16, 120, kernel_size=(5, 5)),
            torch.nn.ReLU(),
            # reshape to [-1, 120]
            torch.nn.Flatten(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 10),
            torch.nn.LogSoftmax(dim=1)
        )
        self.to(self._device)

    def forward(self, images):
        """
        Performs the forward step of the model.

        Args:
            images: Batch of images to be classified.
        """

        predictions = self.model(images)
        return predictions
