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

from nemo.backends.pytorch import LossNM
from nemo.core import ChannelType, LabelsType, LogitsType, LossType, NeuralType

__all__ = ['TokenClassificationLoss']


class TokenClassificationLoss(LossNM):
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
            # "logits": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag)}),
            # "labels": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "loss_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "logits": NeuralType(LogitsType(), ('B', 'T', 'D')),
            "labels": NeuralType(LabelsType(), ('B', 'T')),
            "loss_mask": NeuralType(ChannelType(), ('B', 'T')),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(LossType())}

    def __init__(self, num_classes, class_weights=None):
        LossNM.__init__(self)
        if class_weights:
            class_weights = torch.FloatTensor(class_weights).to(self._device)

        self._criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.num_classes = num_classes

    def _loss_function(self, logits, labels, loss_mask):
        active_loss = loss_mask.view(-1) > 0.5
        active_logits = logits.view(-1, self.num_classes)[active_loss]
        active_labels = labels.view(-1)[active_loss]

        loss = self._criterion(active_logits, active_labels)
        return loss
