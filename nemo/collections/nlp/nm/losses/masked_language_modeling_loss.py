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

from nemo.backends.pytorch import LossNM
from nemo.collections.nlp.nm.losses.smoothed_cross_entropy_loss import SmoothedCrossEntropyLoss
from nemo.core import ChannelType, LogitsType, LossType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['MaskedLanguageModelingLossNM']


class MaskedLanguageModelingLossNM(LossNM):
    """
    Neural module which implements Masked Language Modeling (MLM) loss.

    Args:
        label_smoothing (float): label smoothing regularization coefficient
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "logits": NeuralType(('B', 'T', 'D'), LogitsType()),
            "output_ids": NeuralType(('B', 'T'), ChannelType()),
            "output_mask": NeuralType(('B', 'T'), ChannelType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, label_smoothing=0.0):
        LossNM.__init__(self)
        self._criterion = SmoothedCrossEntropyLoss(label_smoothing)

    def _loss_function(self, logits, output_ids, output_mask):
        loss = self._criterion(logits, output_ids, output_mask)
        return loss
