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

# =============================================================================
# Copyright 2019 Salesforce Research.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom
# the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
# THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# =============================================================================

import torch

from nemo.backends.pytorch.nm import LossNM
from nemo.core.neural_types import LabelsType, LengthsType, LogitsType, LossType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['MaskedLogLoss']


class MaskedLogLoss(LossNM):
    """
    Neural module which implements a cross entropy model with masking feature. It keeps just the target logit for cross entropy calculation

    Args:
        logits (float): output of the classifier
        labels (long): ground truth targets
        loss_mask (long): specifies the ones to get ignored in loss calculation


    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.

        logits: 4d tensor of logits

        labels: 3d tensor of labels

        loss_mask: specifies the words to be considered in the loss calculation

        """
        return {
            "logits": NeuralType(('B', 'T', 'D', 'D'), LogitsType()),
            "labels": NeuralType(('B', 'D', 'T'), LabelsType()),
            "length_mask": NeuralType(('B', 'D'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        # return {"loss": NeuralType(None)}
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self):
        LossNM.__init__(self)

    def _loss_function(self, logits, labels, length_mask, eps=1e-10):
        logits_flat = logits.view(-1, logits.size(-1))
        log_probs_flat = torch.log(torch.clamp(logits_flat, min=eps))
        labels_flat = labels.view(-1, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=labels_flat)
        losses = losses_flat.view(*labels.size())
        loss = self.masking(losses, length_mask)
        return loss

    @staticmethod
    def masking(losses, length_mask):
        max_len = losses.size(2)

        mask_ = torch.arange(max_len, device=length_mask.device)[None, None, :] < length_mask[:, :, None]
        mask_ = mask_.float()
        losses = losses * mask_
        loss = losses.sum() / mask_.sum()
        return loss
