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

import torch
import math

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import *
from nemo.core.neural_types.neural_type import NeuralType

from nemo.utils.decorators import experimental

@experimental
class GlowTTSLoss(Loss):
    """
    Loss for the GlowTTS model
    """

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    @property
    def input_types(self):
        return {
            "z": NeuralType(('B', 'D', 'T'), NormalDistributionSamplesType()),
            "y_m": NeuralType(('B', 'D', 'T'), NormalDistributionMeanType()),
            "y_logs": NeuralType(('B', 'D', 'T'), NormalDistributionLogVarianceType()),
            "logdet": NeuralType(('B',), LogDeterminantType()),
            "logw": NeuralType(('B', 'T'), TokenLogDurationType()),
            "logw_": NeuralType(('B', 'T'), TokenLogDurationType()),
            "x_lengths": NeuralType(('B',), LengthsType()),
            "y_lengths": NeuralType(('B',), LengthsType()),
        }

    @property
    def output_types(self):
        return {"l_mle": NeuralType(elements_type=LossType()),
                "l_length": NeuralType(elements_type=LossType()),
                "logdet": NeuralType(elements_type=VoidType())}

    def __init__(self):
        super().__init__()


    @typecheck()
    def forward(self, z, y_m, y_logs, logdet, logw, logw_, x_lengths, y_lengths):


        logdet = torch.sum(logdet)
        l_mle = 0.5 * math.log(2 * math.pi) + (
            torch.sum(y_logs)
            + 0.5 * torch.sum(torch.exp(-2 * y_logs) * (z - y_m) ** 2)
            - logdet
        ) / (torch.sum(y_lengths) * z.shape[1])

        l_length = torch.sum((logw - logw_) ** 2) / torch.sum(x_lengths)
        return l_mle, l_length, logdet