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

# MIT License
#
# Copyright (c) 2020 Jaehyeon Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import torch

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import *
from nemo.core.neural_types.neural_type import NeuralType


class GlowTTSLoss(Loss):
    """
    Loss for the GlowTTS model
    """

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
        return {
            "l_mle": NeuralType(elements_type=LossType()),
            "l_length": NeuralType(elements_type=LossType()),
            "logdet": NeuralType(elements_type=VoidType()),
        }

    @typecheck()
    def forward(self, z, y_m, y_logs, logdet, logw, logw_, x_lengths, y_lengths):

        logdet = torch.sum(logdet)
        l_mle = 0.5 * math.log(2 * math.pi) + (
            torch.sum(y_logs) + 0.5 * torch.sum(torch.exp(-2 * y_logs) * (z - y_m) ** 2) - logdet
        ) / (torch.sum(y_lengths) * z.shape[1])

        l_length = torch.sum((logw - logw_) ** 2) / torch.sum(x_lengths)
        return l_mle, l_length, logdet
