# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import (
    TokenIndex,
    LengthsType,
    LogitsType,
    LossType,
    PredictionsType,
    RegressionValuesType,
)
from nemo.core.neural_types.neural_type import NeuralType


class MaskedMSELoss(Loss):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss_fn = torch.nn.MSELoss(reduction='none')

    @property
    def input_types(self):
        return {
            "predicted": NeuralType(('B', 'T'), PredictionsType()),
            "target": NeuralType(('B', 'T'), RegressionValuesType()),
            "target_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, predicted, target, target_len):
        assert target.shape[1] == predicted.shape[1]

        # [B, T]
        loss = self.loss_fn(input=predicted, target=target)
        # [B]
        loss = torch.sum(loss, dim=1) / torch.clamp(target_len, min=1.0)
        # [1]
        loss = torch.mean(loss)
        return loss


class MaskedSoftmax(Loss):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()
        self.ignore_index = -1
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='mean')

    @property
    def input_types(self):
        return {
            "logits": NeuralType(('B', 'C', 'T'), LogitsType()),
            "target_index": NeuralType(('B', 'T'), TokenIndex()),
            "target_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, logits, target_index, target_len):
        assert logits.shape[2] == target_index.shape[1]
        mask = get_mask_from_lengths(target_len)
        target = torch.where(mask, target_index.long(), self.ignore_index)
        loss = self.loss_fn(input=logits, target=target)
        return loss


class AudioTokenLoss(Loss):
    def __init__(self, num_codebooks):
        super(AudioTokenLoss, self).__init__()
        self.num_codebooks = num_codebooks
        self.loss_fn = MaskedSoftmax()

    @property
    def input_types(self):
        return {
            "logits": NeuralType(('B', 'C', 'W', 'T'), LogitsType()),
            "target_tokens": NeuralType(('B', 'C', 'T'), TokenIndex()),
            "target_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, logits, target_tokens, target_len):
        assert logits.shape[1] == target_tokens.shape[1] == self.num_codebooks
        assert logits.shape[3] == target_tokens.shape[2]

        loss = 0.0
        for i in range(self.num_codebooks):
            # [B, W, T]
            logits_i = logits[:, i, :, :]
            # [B, T]
            target_i = target_tokens[:, i, :]
            loss += self.loss_fn(logits=logits_i, target_index=target_i, target_len=target_len)

        loss /= self.num_codebooks
        return loss