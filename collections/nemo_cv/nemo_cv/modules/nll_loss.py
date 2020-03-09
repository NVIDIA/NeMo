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


class NLLLoss(LossNM):
    @staticmethod
    def create_ports():
        input_ports = {
            "predictions": NeuralType(
                {0: AxisType(BatchTag),
                 1: AxisType(LogProbabilityTag)}),
            "targets": NeuralType({0: AxisType(BatchTag)}),
        }
        output_ports = {"loss": NeuralType(None)}
        return input_ports, output_ports

    def __init__(self, **kwargs):
        # Neural Module API specific
        LossNM.__init__(self, **kwargs)
        # End of Neural Module API specific
        self._criterion = torch.nn.NLLLoss()

    # You need to implement this function
    def _loss_function(self, **kwargs):
        return self._criterion(*(kwargs.values()))
