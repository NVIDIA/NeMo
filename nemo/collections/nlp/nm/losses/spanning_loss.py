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

from torch import nn

from nemo.backends.pytorch import LossNM
from nemo.core import ChannelType, LogitsType, LossType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['SpanningLoss']


class SpanningLoss(LossNM):
    """
    Neural module which implements QuestionAnswering loss.
    Args:
        logits: Output of question answering head, which is a token classfier.
        start_positions: Ground truth start positions of the answer w.r.t.
            input sequence. If question is unanswerable, this will be
            pointing to start token, e.g. [CLS], of the input sequence.
        end_positions: Ground truth end positions of the answer w.r.t.
            input sequence. If question is unanswerable, this will be
            pointing to start token, e.g. [CLS], of the input sequence.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "logits": NeuralType(('B', 'T', 'D'), LogitsType()),
            "start_positions": NeuralType(tuple('B'), ChannelType()),
            "end_positions": NeuralType(tuple('B'), ChannelType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)

        start_logits:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        end_logits:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)
        """
        return {
            "loss": NeuralType(elements_type=LossType()),
            "start_logits": NeuralType(('B', 'T'), ChannelType()),
            "end_logits": NeuralType(('B', 'T'), ChannelType()),
        }

    def __init__(self):
        LossNM.__init__(self)

    def _loss_function(self, **kwargs):
        logits = kwargs['logits']
        start_positions = kwargs['start_positions']
        end_positions = kwargs['end_positions']
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss, start_logits, end_logits
