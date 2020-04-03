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

from nemo.backends.pytorch import LossNM
from nemo.collections.nlp.utils.data_utils import mask_padded_tokens
from nemo.core import LabelsType, LogitsType, LossType, MaskType, NeuralType

__all__ = ['SmoothedCrossEntropyLoss']


class SmoothedCrossEntropyLoss(LossNM):
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

    def __init__(self, pad_id=None, label_smoothing=0, predict_last_k=0):
        LossNM.__init__(self)

        self._loss_fn = SmoothedCrossEntropy(label_smoothing, predict_last_k)
        self._pad_id = pad_id

    def _loss_function(self, logits, labels, output_mask=None):
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
