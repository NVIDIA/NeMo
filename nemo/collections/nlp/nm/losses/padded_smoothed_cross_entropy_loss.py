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

from nemo.backends.pytorch import LossNM
from nemo.collections.nlp.nm.losses.smoothed_cross_entropy_loss import SmoothedCrossEntropyLoss
from nemo.collections.nlp.utils.common_nlp_utils import mask_padded_tokens
from nemo.core import AxisType, ChannelType, LogitsType, LossType, NeuralType

__all__ = ['PaddedSmoothedCrossEntropyLossNM']


class PaddedSmoothedCrossEntropyLossNM(LossNM):
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
            # "logits": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag)}),
            # "target_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "logits": NeuralType(LogitsType(), ('B', 'T', 'D')),
            "target_ids": NeuralType(ChannelType(), ('B', 'T')),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        # return {"loss": NeuralType(None)}
        return {"loss": NeuralType(LossType())}

    def __init__(self, pad_id, label_smoothing=0, predict_last_k=0):
        LossNM.__init__(self)

        self._loss_fn = SmoothedCrossEntropyLoss(label_smoothing, predict_last_k)
        self._pad_id = pad_id

    def _loss_function(self, logits, target_ids):
        target_mask = mask_padded_tokens(target_ids, self._pad_id).to(logits.dtype)
        loss = self._loss_fn(logits, target_ids, target_mask)
        return loss
