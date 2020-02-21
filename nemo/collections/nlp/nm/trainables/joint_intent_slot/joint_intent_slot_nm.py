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

from torch import nn as nn

from nemo.backends.pytorch import MultiLayerPerceptron, TrainableNM
from nemo.collections.nlp.utils.transformer_utils import transformer_weights_init
from nemo.core import ChannelType, LogitsType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['JointIntentSlotClassifier']


class JointIntentSlotClassifier(TrainableNM):
    """
    The softmax classifier for the joint intent classification and slot
    filling task which  consists of a dense layer + relu + softmax for
    predicting the slots and similar for predicting the intents.

    Args:
        hidden_size (int): the size of the hidden state for the dense layer
        num_intents (int): number of intents
        num_slots (int): number of slots
        dropout (float): dropout to be applied to the layer
        use_transformer_pretrained (bool):
            TODO
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.

        hidden_states:
            TODO
        """
        return {"hidden_states": NeuralType(('B', 'T', 'C'), ChannelType())}

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        intent_logits:
            TODO
        slot_logits:
            TODO
        """
        return {
            "intent_logits": NeuralType(('B', 'D'), LogitsType()),
            "slot_logits": NeuralType(('B', 'T', 'D'), LogitsType()),
        }

    def __init__(self, hidden_size, num_intents, num_slots, dropout=0.0, use_transformer_pretrained=True, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.slot_mlp = MultiLayerPerceptron(
            hidden_size, num_classes=num_slots, device=self._device, num_layers=2, activation='relu', log_softmax=False
        )
        self.intent_mlp = MultiLayerPerceptron(
            hidden_size,
            num_classes=num_intents,
            device=self._device,
            num_layers=2,
            activation='relu',
            log_softmax=False,
        )
        if use_transformer_pretrained:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        intent_logits = self.intent_mlp(hidden_states[:, 0])
        slot_logits = self.slot_mlp(hidden_states)
        return intent_logits, slot_logits
