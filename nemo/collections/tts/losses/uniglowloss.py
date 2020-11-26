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

from nemo.collections.tts.losses.stftlosses import MultiResolutionSTFTLoss
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import AudioSignal, LossType, NormalDistributionSamplesType, VoidType
from nemo.core.neural_types.neural_type import NeuralType


class UniGlowLoss(Loss):
    """A Loss module that computes loss for UniGlow"""

    def __init__(self, stft_loss_coef):
        super(UniGlowLoss, self).__init__()
        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window="hann_window"
        )
        self.stft_loss_coef = stft_loss_coef

    @property
    def input_types(self):
        return {
            "z": NeuralType(('B', 'flowgroup', 'T'), NormalDistributionSamplesType()),
            "logdet": NeuralType(elements_type=VoidType()),
            "gt_audio": NeuralType(('B', 'T'), AudioSignal()),
            "predicted_audio": NeuralType(('B', 'T'), AudioSignal()),
            "sigma": NeuralType(optional=True),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, *, z, logdet, gt_audio, predicted_audio, sigma=1.0):
        nll_loss = torch.sum(z * z) / (2 * sigma * sigma) - logdet
        nll_loss = nll_loss / (z.size(0) * z.size(1) * z.size(2))
        sc_loss, mag_loss = self.stft_loss(predicted_audio, gt_audio)
        sc_loss = sum(sc_loss) / len(sc_loss)
        mag_loss = sum(mag_loss) / len(mag_loss)
        stft_loss = sc_loss + mag_loss
        loss = nll_loss + self.stft_loss_coef * stft_loss
        return loss
