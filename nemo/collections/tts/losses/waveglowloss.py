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

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import LossType, NormalDistributionSamplesType, VoidType
from nemo.core.neural_types.neural_type import NeuralType


class WaveGlowLoss(Loss):
    """ A Loss module that computes loss for WaveGlow
    """

    @property
    def input_types(self):
        return {
            "z": NeuralType(('B', 'flowgroup', 'T'), NormalDistributionSamplesType()),
            "log_s_list": [NeuralType(('B', 'flowgroup', 'T'), VoidType())],  # TODO: Figure out a good typing
            "log_det_W_list": [NeuralType(elements_type=VoidType())],  # TODO: Figure out a good typing
            "sigma": NeuralType(optional=True),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, *, z, log_s_list, log_det_W_list, sigma=1.0):
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]

        loss = torch.sum(z * z) / (2 * sigma * sigma) - log_s_total - log_det_W_total
        return loss / (z.size(0) * z.size(1) * z.size(2))
