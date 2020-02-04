# =============================================================================
# Copyright 2019 AI Applications Design Team at NVIDIA. All Rights Reserved.
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
from nemo.core.neural_types import AxisType, BatchTag, ChannelTag, NeuralType, TimeTag

__all__ = ['TRADEMaskedCrosEntropy', 'CrossEntropyLoss3D']


class TRADEMaskedCrosEntropy(LossNM):
    """
    Neural module which implements Masked Language Modeling (MLM) loss.

    Args:
        label_smoothing (float): label smoothing regularization coefficient
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        hidden_states:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

            2: AxisType(ChannelTag)
        """
        return {
            "logits": NeuralType(
                {0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag), 3: AxisType(ChannelTag)}
            ),
            "targets": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag), 2: AxisType(TimeTag)}),
            "mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        intent_logits:
            0: AxisType(BatchTag)

            1: AxisType(ChannelTag)

        slot_logits:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

            2: AxisType(ChannelTag)
        """
        return {"loss": NeuralType(None)}

    def __init__(self):
        LossNM.__init__(self)

    def _loss_function(self, logits, targets, mask):
        logits_flat = logits.view(-1, logits.size(-1))
        eps = 1e-10
        log_probs_flat = torch.log(torch.clamp(logits_flat, min=eps))
        target_flat = targets.view(-1, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        losses = losses_flat.view(*targets.size())
        loss = self.masking(losses, mask)
        return loss

    @staticmethod
    def masking(losses, mask):

        # mask_ = []
        # batch_size = mask.size(0)
        max_len = losses.size(2)

        mask_ = torch.arange(max_len, device=mask.device)[None, None, :] < mask[:, :, None]
        # mask_ = torch.arange(max_len, device=mask.device).expand(losses.size()) < mask.expand(losses)

        # for si in range(mask.size(1)):
        #     seq_range = torch.arange(0, max_len).long()
        #     seq_range_expand = \
        #         seq_range.unsqueeze(0).expand(batch_size, max_len)
        #     if mask[:, si].is_cuda:
        #         seq_range_expand = seq_range_expand.cuda()
        #     seq_length_expand = mask[:, si].unsqueeze(
        #         1).expand_as(seq_range_expand)
        #     mask_.append((seq_range_expand < seq_length_expand))
        # mask_ = torch.stack(mask_)
        # mask_ = mask_.transpose(0, 1)
        # if losses.is_cuda:
        #     mask_ = mask_.cuda()

        mask_ = mask_.float()
        losses = losses * mask_
        loss = losses.sum() / mask_.sum()
        return loss


class CrossEntropyLoss3D(LossNM):
    """
    Neural module which implements Token Classification loss.
    Args:
        num_classes (int): number of classes in a classifier, e.g. size
            of the vocabulary in language modeling objective
        logits (float): output of the classifier
        labels (long): ground truth labels
        loss_mask (long): to differentiate from original tokens and paddings
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "logits": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag), 2: AxisType(ChannelTag)}),
            "labels": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {"loss": NeuralType(None)}

    def __init__(self, num_classes, **kwargs):
        LossNM.__init__(self, **kwargs)
        self._criterion = torch.nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def _loss_function(self, logits, labels):
        logits_flatten = logits.view(-1, self.num_classes)
        labels_flatten = labels.view(-1)

        loss = self._criterion(logits_flatten, labels_flatten)
        return loss
