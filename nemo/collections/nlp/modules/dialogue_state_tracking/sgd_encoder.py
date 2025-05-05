# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, Optional

from torch import nn as nn

from nemo.collections.nlp.modules.common.classifier import Classifier
from nemo.core.classes import typecheck
from nemo.core.neural_types import ChannelType, LogitsType, NeuralType

__all__ = ['SGDEncoder']

ACT2FN = {"tanh": nn.functional.tanh, "relu": nn.functional.relu}


class SGDEncoder(Classifier):
    """
    Neural module which encodes BERT hidden states
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns definitions of module output ports.
        """

        return {
            "logits": NeuralType(('B', 'T'), LogitsType()),
            'hidden_states': NeuralType(('B', 'T', 'C'), ChannelType()),
        }

    def __init__(
        self, hidden_size: int, activation: str = 'tanh', dropout: float = 0.0, use_transformer_init: bool = True,
    ) -> None:

        """
        Args:
            hidden_size: hidden size of the BERT model
            activation: activation function applied
            dropout: dropout ratio
            use_transformer_init: use transformer initialization
        """
        super().__init__(hidden_size=hidden_size, dropout=dropout)
        self.fc = nn.Linear(hidden_size, hidden_size)

        if activation not in ACT2FN:
            raise ValueError(f'{activation} is not in supported ' + '{ACTIVATIONS_F.keys()}')

        self.activation = ACT2FN[activation]
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.post_init(use_transformer_init=use_transformer_init)

    @typecheck()
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: bert output hidden states
        """
        first_token_hidden_states = hidden_states[:, 0]
        logits = self.fc(first_token_hidden_states)
        logits = self.activation(logits)
        logits = self.dropout1(logits)
        return logits, self.dropout2(hidden_states)
