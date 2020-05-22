# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
# Copyright 2019 The Google Research Authors.
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

'''
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst/baseline/train_and_predict.py
'''

from torch import nn

from nemo.backends.pytorch.nm import TrainableNM
from nemo.collections.nlp.utils.transformer_utils import transformer_weights_init
from nemo.core import ChannelType, EmbeddedTextType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['SGDEncoderNM']

ACTIVATIONS_F = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
}


class SGDEncoderNM(TrainableNM):
    """
    Neural module which extracts the first token from the BERT representation of the utterance
    followed by a fully connected layer.

    Args:
        hidden_size (int): hidden size of the BERT model
        activation (str): activation function applied
        dropout (float): dropout ratio
    """

    @property
    @add_port_docs
    def input_ports(self):
        """
        Returns definitions of module input ports.
        hidden_states (float): BERT representation of the utterance
        """
        return {"hidden_states": NeuralType(('B', 'T', 'C'), ChannelType())}

    @property
    @add_port_docs
    def output_ports(self):
        """Returns definitions of module output ports.
        logits (float): First token of the BERT representation of the utterance followed by fc and dropout
        hidden_states (float) : BERT representation of the utterance with applied dropout
        """
        return {
            "logits": NeuralType(('B', 'T'), EmbeddedTextType()),
            "hidden_states": NeuralType(('B', 'T', 'C'), ChannelType()),
        }

    def __init__(self, hidden_size, activation='tanh', dropout=0.0, use_transformer_pretrained=True):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size).to(self._device)

        if activation not in ACTIVATIONS_F:
            raise ValueError(f'{activation} is not in supported ' + '{ACTIVATIONS_F.keys()}')

        self.activation = ACTIVATIONS_F[activation]()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if use_transformer_pretrained:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))
        # self.to(self._device) # sometimes this is necessary

    def forward(self, hidden_states):
        first_token_hidden_states = hidden_states[:, 0]
        logits = self.fc(first_token_hidden_states)
        logits = self.activation(logits)
        logits = self.dropout1(logits)
        return logits, self.dropout2(hidden_states)
