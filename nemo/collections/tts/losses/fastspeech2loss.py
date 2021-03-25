# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import LengthsType, LossType, MelSpectrogramType
from nemo.core.neural_types.neural_type import NeuralType


class L1MelLoss(Loss):
    """A Loss module that computes L1 mel loss for FastSpeech 2."""

    @property
    def input_types(self):
        return {
            "spec_pred": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "spec_target": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "spec_target_len": NeuralType(('B'), LengthsType()),
            "pad_value": NeuralType(),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, *, spec_pred, spec_target, spec_target_len, pad_value, transpose=True):
        if transpose:
            spec_pred = spec_pred.transpose(1, 2)
        spec_target.requires_grad = False
        max_len = spec_target.shape[2]

        if max_len < spec_pred.shape[2]:
            # Predicted len is larger than reference
            # Need to slice
            spec_pred = spec_pred.narrow(2, 0, max_len)
        elif max_len > spec_pred.shape[2]:
            # Need to do padding
            pad_amount = max_len - spec_pred.shape[2]
            spec_pred = torch.nn.functional.pad(spec_pred, (0, pad_amount), value=pad_value)
            max_len = spec_pred.shape[2]

        mask = ~get_mask_from_lengths(spec_target_len, max_len=max_len)
        mask = mask.expand(spec_target.shape[1], mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2)
        spec_pred.masked_fill_(mask, pad_value)

        mel_loss = torch.nn.functional.l1_loss(spec_pred, spec_target)
        return mel_loss


class L2MelLoss(Loss):
    """A Loss module that computes L2 mel loss for FastSpeech 2"""

    @property
    def input_types(self):
        return {
            "spec_pred": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "spec_target": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "spec_target_len": NeuralType(('B'), LengthsType()),
            "pad_value": NeuralType(),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, *, spec_pred, spec_target, spec_target_len, pad_value):
        spec_pred = spec_pred.transpose(1, 2)
        spec_target.requires_grad = False
        max_len = spec_target.shape[2]

        if max_len < spec_pred.shape[2]:
            # Predicted len is larger than reference
            # Need to slice
            spec_pred = spec_pred.narrow(2, 0, max_len)
        elif max_len > spec_pred.shape[2]:
            # Need to do padding
            pad_amount = max_len - spec_pred.shape[2]
            spec_pred = torch.nn.functional.pad(spec_pred, (0, pad_amount), value=pad_value)
            max_len = spec_pred.shape[2]

        mask = ~get_mask_from_lengths(spec_target_len, max_len=max_len)
        mask = mask.expand(spec_target.shape[1], mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2)
        spec_pred.masked_fill_(mask, pad_value)

        mel_loss = torch.nn.functional.mse_loss(spec_pred, spec_target)
        return mel_loss
