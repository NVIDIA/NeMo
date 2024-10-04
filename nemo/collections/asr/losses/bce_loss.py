# ! /usr/bin/python
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from nemo.core.neural_types import LabelsType, LengthsType, LossType, NeuralType, ProbsType

__all__ = ['BCELoss']


class BCELoss(Loss, Typing):
    """
    Computes Binary Cross Entropy (BCE) loss. The BCELoss class expects output from Sigmoid function.
    """

    @property
    def input_types(self):
        """Input types definitions for AnguarLoss.
        """
        return {
            "probs": NeuralType(('B', 'T', 'C'), ProbsType()),
            'labels': NeuralType(('B', 'T', 'C'), LabelsType()),
            "target_lens": NeuralType(('B', 'C'), LengthsType()),
        }

    @property
    def output_types(self):
        """
        Output types definitions for binary cross entropy loss. Weights for labels can be set using weight variables.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, reduction='mean', alpha=1.0, weight=torch.tensor([0.1, 0.9]), sorted_preds: bool=False, sorted_loss: bool=False, class_normalization: bool=False):
        super().__init__()
        self.class_normalization = class_normalization
        if class_normalization:
            self.reduction = 'none'
        else:
            self.reduction = 'mean'
        self.loss_weight = weight
        self.loss_f = torch.nn.BCELoss(reduction=self.reduction)
        self.sorted_preds = sorted_preds
        self.sorted_loss = sorted_loss
        self.eps = 1e-6

    @typecheck()
    def forward(self, probs, labels, target_lens):
        """
        Calculate binary cross entropy loss based on probs, labels and target_lens variables.

        Args:
            probs (torch.tensor)
                Predicted probability value which ranges from 0 to 1. Sigmoid output is expected.
            labels (torch.tensor)
                Groundtruth label for the predicted samples.
            target_lens (torch.tensor):
                The actual length of the sequence without zero-padding.

        Returns:
            loss (NeuralType)
                Binary cross entropy loss value.
        """
        probs_list = [probs[k, : target_lens[k], :] for k in range(probs.shape[0])]
        targets_list = [labels[k, : target_lens[k], :] for k in range(labels.shape[0])]
        probs = torch.cat(probs_list, dim=0)
        labels = torch.cat(targets_list, dim=0)
        if self.class_normalization in ['class', 'class_binary', 'binary']:
            if self.class_normalization in ['class', 'class_binary']:
                # Normalize loss by number of classes
                norm_weight = 1/(labels.sum(dim=0) + self.eps)
                norm_weight_norm = norm_weight / norm_weight.sum()
                norm_weight_norm2 = torch.clamp(norm_weight_norm, min=0.05, max=1.0) 
                norm_weight_norm2 = norm_weight_norm2 / norm_weight_norm2.max()
                norm_weight = norm_weight_norm2[None, :].expand_as(labels).detach().clone()
            else:            
                norm_weight = torch.ones_like(labels).detach().clone()

            if self.class_normalization in ['binary', 'class_binary']:
                binary_weight = torch.ones_like(labels).detach().clone()
                one_weight = (labels.sum() / (labels.shape[0]*labels.shape[1])).to(labels.device)
                binary_weight[labels == 0] = one_weight
                binary_weight[labels == 1] = 1 - one_weight
            else:
                binary_weight = torch.ones_like(labels).detach().clone()
                
        elif self.class_normalization == 'none' or not self.class_normalization:
            binary_weight = torch.ones_like(labels).detach().clone() 
            norm_weight = torch.ones_like(labels).detach().clone()
            
        if self.reduction == 'sum':
            return self.loss_f(probs, labels)
        elif self.reduction == 'mean':
            return self.loss_f(probs, labels).mean()
        elif self.reduction == 'none':
            if self.class_normalization in ['class', 'class_binary', 'binary']:
                return (binary_weight * norm_weight * self.loss_f(probs, labels)).sum()
            else:
                return self.loss_f(probs, labels)
            
