# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

import torch
from torch import nn

from nemo.core.classes import NeuralModule, Typing, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogitsType, LogprobsType, LossType, MaskType, NeuralType
from nemo.utils.decorators import experimental

__all__ = ['SmoothedCrossEntropyLoss', 'CrossEntropyLoss']


def mask_padded_tokens(tokens, pad_id):
    mask = tokens != pad_id
    return mask


# @experimental
class CrossEntropyLoss(NeuralModule):
    """
    CrossEntropyLoss
    Args:
        logits_ndim (int): number of dimensions (or rank) of the logits tensor
        weight (list): list of rescaling weight given to each class
        reduction (str): type of the reduction over the batch
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "logits": NeuralType(['B'] + ['ANY'] * (self._logits_dim - 1), LogitsType()),
            "labels": NeuralType(['B'] + ['ANY'] * (self._logits_dim - 2), LabelsType()),
            "loss_mask": NeuralType(['B'] + ['ANY'] * (self._logits_dim - 2), MaskType(), optional=True),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, logits_ndim=2, weight=None, reduction='mean'):
        super().__init__()

        self._criterion = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self._logits_dim = logits_ndim

    @typecheck()
    def forward(self, logits, labels, loss_mask=None):
        """
        Args:
            logits (float): output of the classifier
            labels (long): ground truth labels
            loss_mask (bool/float/int): tensor to specify the masking
        """
        logits_flatten = torch.flatten(logits, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(labels, start_dim=0, end_dim=-1)

        if loss_mask is not None:
            if loss_mask.dtype is not torch.bool:
                loss_mask = loss_mask > 0.5
            loss_mask_flatten = torch.flatten(loss_mask, start_dim=0, end_dim=-1)
            logits_flatten = logits_flatten[loss_mask_flatten]
            labels_flatten = labels_flatten[loss_mask_flatten]

        if len(labels_flatten) == 0:
            return self._criterion(logits, torch.argmax(logits, dim=-1))

        loss = self._criterion(logits_flatten, labels_flatten)
        return loss

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass


# @experimental
class SmoothedCrossEntropyLoss(NeuralModule):
    """
    Neural module which calculates CrossEntropyLoss and
    1) excludes padding tokens from loss calculation
    2) allows to use label smoothing regularization
    3) allows to calculate loss for the desired number of last tokens
    Args:
        label_smoothing (float): label smoothing regularization coefficient
        predict_last_k (int): how many last tokens to use for the loss
            calculation, important for fast evaluation of LM perplexity
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "logits": NeuralType(('B', 'T', 'D'), LogitsType()),
            "labels": NeuralType(('B', 'T'), LabelsType()),
            "output_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        # return {"loss": NeuralType(None)}
        return {"loss": NeuralType(elements_type=LossType())}

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    def __init__(self, pad_id=None, label_smoothing=0, predict_last_k=0):
        super().__init__()

        self._loss_fn = SmoothedCrossEntropy(label_smoothing, predict_last_k)
        self._pad_id = pad_id

    @typecheck()
    def forward(self, logits, labels, output_mask=None):
        if output_mask is not None:
            labels_mask = output_mask
        elif self._pad_id is not None:
            labels_mask = mask_padded_tokens(labels, self._pad_id).to(logits.dtype)
        else:
            raise ValueError("Both output_mask and pad_id are None")

        if labels_mask.dtype is not logits.dtype:
            labels_mask = labels_mask.to(logits.dtype)

        loss = self._loss_fn(logits, labels, labels_mask)
        return loss


class SmoothedCrossEntropy(torch.nn.Module):
    """
    Cross-entropy loss with label smoothing for a batch of sequences.
    Args:
        label_smoothing (float): label smoothing coefficient, usually set
                                 between 0.0 and 0.1 in language modeling
                                 and translation pipelines
        predict_last_k (int): int parameter which sets the number of last
                              tokens to calculate the loss for, for example
            0: (default) calculate loss on the entire sequence (e.g., NMT)
            1: calculate loss on the last token only (e.g., LM evaluation)
            Intermediate values allow to control the trade-off between eval
            time (proportional to the number of batches) and eval performance
            (proportional to the number of context tokens).
    """

    def __init__(self, label_smoothing=0.0, predict_last_k=0):
        super().__init__()
        self._smoothing = label_smoothing
        self._predict_last_k = predict_last_k

    def forward(self, logits, labels, output_mask, eps=1e-6):
        """
        Args:
            logits: float tensor of shape batch_size x seq_len x vocab_size, values should be log probabilities
            labels: int tensor of shape batch_size x seq_len
            output_mask: binary tensor of shape batch_size x seq_len
            eps: epsilon param to avoid divide by zero in loss calculation
        """
        batch_size, seq_len, vocab_size = logits.size()
        smoothing = vocab_size * self._smoothing / (vocab_size - 1)
        target_logits = logits.gather(2, labels.unsqueeze(2)).squeeze(2)
        smoothing_logits = logits.mean(dim=-1)
        neg_log_likelihood = (1.0 - smoothing) * target_logits + smoothing * smoothing_logits
        neg_log_likelihood = neg_log_likelihood[:, -self._predict_last_k :]
        output_mask = output_mask[:, -self._predict_last_k :]
        neg_log_likelihood = -torch.sum(neg_log_likelihood * output_mask)
        neg_log_likelihood = neg_log_likelihood / (output_mask.sum() + eps)
        return neg_log_likelihood
