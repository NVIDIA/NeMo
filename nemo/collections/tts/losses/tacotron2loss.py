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

from torch import nn
from torch.nn.functional import pad

from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import LengthsType, LogitsType, LossType, MelSpectrogramType
from nemo.core.neural_types.neural_type import NeuralType


class Tacotron2Loss(Loss):
    """ A Loss module that computes loss for Tacotron2
    """

    @property
    def input_types(self):
        return {
            "mel_out": NeuralType(('B', 'T', 'D'), MelSpectrogramType()),
            "mel_out_postnet": NeuralType(('B', 'T', 'D'), MelSpectrogramType()),
            "gate_out": NeuralType(('B', 'T'), LogitsType()),
            "mel_target": NeuralType(('B', 'T', 'D'), MelSpectrogramType()),
            "gate_target": NeuralType(('B', 'T'), LogitsType()),
            "target_len": NeuralType(('B'), LengthsType()),
            "pad_value": NeuralType(),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, *, mel_out, mel_out_postnet, gate_out, mel_target, gate_target, target_len, pad_value):
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        max_len = mel_target.shape[2]

        if max_len < mel_out.shape[2]:
            # Predicted len is larger than reference
            # Need to slice
            mel_out = mel_out.narrow(2, 0, max_len)
            mel_out_postnet = mel_out_postnet.narrow(2, 0, max_len)
            gate_out = gate_out.narrow(1, 0, max_len).contiguous()
        elif max_len > mel_out.shape[2]:
            # Need to do padding
            pad_amount = max_len - mel_out.shape[2]
            mel_out = pad(mel_out, (0, pad_amount), value=pad_value)
            mel_out_postnet = pad(mel_out_postnet, (0, pad_amount), value=pad_value)
            gate_out = pad(gate_out, (0, pad_amount), value=1e3)
            max_len = mel_out.shape[2]

        mask = ~get_mask_from_lengths(target_len, max_len=max_len)
        mask = mask.expand(mel_target.shape[1], mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2)
        mel_out.data.masked_fill_(mask, pad_value)
        mel_out_postnet.data.masked_fill_(mask, pad_value)
        gate_out.data.masked_fill_(mask[:, 0, :], 1e3)

        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss
