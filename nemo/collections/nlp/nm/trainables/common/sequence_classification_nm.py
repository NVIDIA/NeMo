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

__all__ = ['SequenceClassifier']


class SequenceClassifier(TrainableNM):
    """
    Neural module which consists of MLP followed by softmax classifier for each
    sequence in the batch.

    Args:
        hidden_size (int): hidden size (d_model) of the Transformer
        num_classes (int): number of classes in softmax classifier, e.g. number
            of different sentiments
        num_layers (int): number of layers in classifier MLP
        activation (str): activation function applied in classifier MLP layers
        log_softmax (bool): whether to apply log_softmax to MLP output
        dropout (float): dropout ratio applied to MLP
        use_transformer_pretrained (bool):
            TODO
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        hidden_states: embedding hidden states
        """
        return {"hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        logits: logits before loss
        """
        return {"logits": NeuralType(('B', 'D'), LogitsType())}

    def __init__(
        self,
        hidden_size,
        num_classes,
        num_layers=2,
        activation='relu',
        log_softmax=True,
        dropout=0.0,
        use_transformer_pretrained=True,
    ):
        super().__init__()
        self.mlp = MultiLayerPerceptron(hidden_size, num_classes, self._device, num_layers, activation, log_softmax)
        self.dropout = nn.Dropout(dropout)
        if use_transformer_pretrained:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))
        # self.to(self._device) # sometimes this is necessary

    def forward(self, hidden_states, idx_conditioned_on=0):
        hidden_states = self.dropout(hidden_states)
        logits = self.mlp(hidden_states[:, idx_conditioned_on])
        return logits
