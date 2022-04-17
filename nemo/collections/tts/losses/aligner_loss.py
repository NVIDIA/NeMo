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
import torch.nn.functional as F

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import LengthsType, LogprobsType, LossType, ProbsType
from nemo.core.neural_types.neural_type import NeuralType


class ForwardSumLoss(Loss):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    @property
    def input_types(self):
        return {
            "attn_logprob": NeuralType(('B', 'S', 'T_spec', 'T_text'), LogprobsType()),
            "in_lens": NeuralType(tuple('B'), LengthsType()),
            "out_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "forward_sum_loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss


class BinLoss(Loss):
    def __init__(self):
        super().__init__()

    @property
    def input_types(self):
        return {
            "hard_attention": NeuralType(('B', 'S', 'T_spec', 'T_text'), ProbsType()),
            "soft_attention": NeuralType(('B', 'S', 'T_spec', 'T_text'), ProbsType()),
        }

    @property
    def output_types(self):
        return {
            "bin_loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()
        return -log_sum / hard_attention.sum()
