# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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

from nemo.collections.common.parts.multi_layer_perceptron import MultiLayerPerceptron
from nemo.collections.common.parts.transformer_utils import transformer_weights_init
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import ChannelType, LogitsType, NeuralType
from nemo.utils.decorators import experimental


@experimental
class SequenceClassifier(NeuralModule):
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
    # @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        hidden_states: embedding hidden states
        """
        return {"hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    @property
    # @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        logits: logits before loss
        """
        return {"logits": NeuralType(('B', 'D'), LogitsType())}

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        activation: str = 'relu',
        log_softmax: bool = True,
        dropout: float = 0.0,
        use_transformer_init: bool = True,
    ):
        """
        Initializes the SequenceClassifier module.
        Args:
            hidden_size (int): the hidden size of the mlp head on the top of the encoder
            num_classes (int): number of the classes to predict
            num_layers (int)_layers (int): number of the linear layers of the mlp head on the top of the encoder
            activation (str): type of activations between layers of the mlp head
            log_softmax (bool): applies the log softmax on the output
            dropout (float): the dropout used for the mlp head
            use_transformer_init (bool): initializes the weights with the same approach used in Transformer
        """
        super().__init__()
        # TODO: what happens to device?
        self.mlp = MultiLayerPerceptron(
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            activation=activation,
            log_softmax=log_softmax,
        )
        self.dropout = torch.nn.Dropout(dropout)
        if use_transformer_init:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))
        # TODO: what happens to device?
        # self.to(self._device) # sometimes this is necessary

    @typecheck()
    def forward(self, hidden_states, idx_conditioned_on=0):
        hidden_states = self.dropout(hidden_states)
        logits = self.mlp(hidden_states[:, idx_conditioned_on])
        return logits

    @classmethod
    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass
