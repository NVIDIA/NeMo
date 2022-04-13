# ! /usr/bin/python
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

from nemo.core.classes import Loss, Typing, typecheck
from nemo.core.neural_types import LabelsType, LogprobsType, NeuralType, LossType

__all__ = ['BCELoss']


class BCELoss(Loss, Typing):
    """
    Computes Binary Cross Entropy (BCE) loss.
    """

    @property
    def input_types(self):
        """Input types definitions for AnguarLoss.
        """
        return {
            "logits": NeuralType(('B', 'T', 'C'), LogprobsType()),
            'labels': NeuralType(('B', 'T', 'C'), LabelsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for AngularLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, reduction='sum', alpha=1.0, weight=torch.tensor([0.1, 0.9])):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = weight
        self.alpha = alpha
        self.loss_f = torch.nn.BCELoss(reduction=self.reduction)

    @typecheck()
    def forward(self, logits, labels):
        # logits.requires_grad = True
        self.positive = logits.round().bool() == 1
        self.negative = logits.round().bool() == 0
        self.positive_label = labels.round().bool() == 1
        self.negative_label = labels.round().bool() == 0

        # print("logits:", logits, "labels:", labels)
        # print("logits type:", type(logits), "labels: type", type(labels))
        # print("logits.shape:", logits.shape, "labels.shape:", labels.shape)
        self.true = logits.round().bool() == labels.round().bool()
        self.false = logits.round().bool() != labels.round().bool()
        # self.true_positive_count = torch.sum(self.true == self.positive)
        # self.false_positive_count = torch.sum(self.false == self.positive)
        self.true_positive_count = torch.sum(torch.logical_and(self.true, self.positive))
        self.false_positive_count = torch.sum(torch.logical_and(self.false, self.positive))
        self.total_counts_k = torch.prod(torch.tensor(labels.shape))
        # print("numer: ", torch.sum(self.positive_label))
        # print("denom: ", self.total_counts_k)
        self.ground_truth_pos_rate = torch.tensor(torch.sum(self.positive_label) / self.total_counts_k, requires_grad=False)
        min_len = min(logits.shape[1], labels.shape[1])
        logits, labels = logits[:, :min_len, :], labels[:, :min_len, :]
        weight = torch.clone(labels)
        weight[weight == 0] = self.loss_weight[0]
        weight[weight == 1] = self.loss_weight[1]
        # loss_f = torch.nn.BCELoss(reduction=self.reduction, weight=weight)
        # print("logits:", logits)
        # self.positive = torch.sum(logits.round().bool() == True)
        # self.total_counts_k = torch.prod(torch.tensor(labels.shape))
        # print("[LOSS] self.ground_truth_pos_rate:", self.ground_truth_pos_rate)
        # return loss_f(logits, labels) + self.alpha * self.ground_truth_pos_rate
        return self.loss_f(logits, labels) 

