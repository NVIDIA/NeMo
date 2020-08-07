# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from torch import nn as nn

from nemo.collections.common.parts import MultiLayerPerceptron, transformer_weights_init
from nemo.core.classes import Exportable, NeuralModule, typecheck
from nemo.core.neural_types import ChannelType, LogitsType, NeuralType

__all__ = ['BertPretrainingTokenClassifier', 'TokenClassifier']

ACT2FN = {"gelu": nn.functional.gelu, "relu": nn.functional.relu}


class TokenClassifier(NeuralModule, Exportable):
    """
    A module to perform token level classification tasks such as Named entity recognition.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns definitions of module input ports.
        """
        return {"hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns definitions of module output ports.
        """
        return {"logits": NeuralType(('B', 'T', 'C'), LogitsType())}

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        activation: str = 'relu',
        log_softmax: bool = True,
        dropout: float = 0.0,
        use_transformer_init: bool = True,
    ) -> None:

        """
        Initializes the Token Classifier module.

        Args:
            hidden_size: the size of the hidden dimension
            num_classes: number of classes
            num_layers: number of fully connected layers in the multilayer perceptron (MLP)
            activation: activation to usee between fully connected layers in the MLP
            log_softmax: whether to apply softmax to the output of the MLP
            dropout: dropout to apply to the input hidden states
            use_transformer_init: whether to initialize the weights of the classifier head with the same approach used in Transformer
        """
        super().__init__()
        self.mlp = MultiLayerPerceptron(
            hidden_size, num_classes, num_layers=num_layers, activation=activation, log_softmax=log_softmax
        )
        self._hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        if use_transformer_init:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))

    @typecheck()
    def forward(self, hidden_states):
        """
        Performs the forward step of the module.
        Args:
            hidden_states: batch of hidden states (for example, from the BERT encoder module)
                [BATCH_SIZE x SEQ_LENGTH x HIDDEN_SIZE]
        Returns: logits value for each class [BATCH_SIZE x SEQ_LENGTH x NUM_CLASSES]
        """
        hidden_states = self.dropout(hidden_states)
        logits = self.mlp(hidden_states)
        return logits

    def _prepare_for_export(self):
        """
        Returns a pair in input, output examples for tracing.
        Returns:
            A pair of (input, output) examples.
        """
        bs = 8
        seq = 64
        input_example = torch.randn(bs, seq, self._hidden_size).to(next(self.parameters()).device)
        output_example = self.forward(hidden_states=input_example)
        return input_example, output_example

    def save_to(self, save_path: str):
        """
        Saves the module to the specified path.
        Args:
            save_path: Path to where to save the module.
        """
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """
        Restores the module from the specified path.
        Args:
            restore_path: Path to restore the module from.
        """
        pass


class BertPretrainingTokenClassifier(NeuralModule, Exportable):
    """
    A module to perform token level classification tasks for Bert pretraining.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns definitions of module input ports.
        """
        return {"hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns definitions of module output ports.
        """
        return {"logits": NeuralType(('B', 'T', 'C'), LogitsType())}

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        activation: str = 'relu',
        log_softmax: bool = True,
        dropout: float = 0.0,
        use_transformer_init: bool = True,
    ) -> None:

        """
        Initializes the Token Classifier module.

        Args:
            hidden_size: the size of the hidden dimension
            num_classes: number of classes
            num_layers: number of fully connected layers in the multilayer perceptron (MLP)
            activation: activation to usee between fully connected layers in the MLP
            log_softmax: whether to apply softmax to the output of the MLP
            dropout: dropout to apply to the input hidden states
            use_transformer_init: whether to initialize the weights of the classifier head with the same approach used in Transformer
        """
        super().__init__()
        self._hidden_size = hidden_size
        if activation not in ACT2FN:
            raise ValueError(f'activation "{activation}" not found')
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = ACT2FN[activation]
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.mlp = MultiLayerPerceptron(
            hidden_size, num_classes, num_layers=num_layers, activation=activation, log_softmax=log_softmax
        )
        self.dropout = nn.Dropout(dropout)
        if use_transformer_init:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))

    @typecheck()
    def forward(self, hidden_states):
        """
        Performs the forward step of the module.
        Args:
            hidden_states: batch of hidden states (for example, from the BERT encoder module)
                [BATCH_SIZE x SEQ_LENGTH x HIDDEN_SIZE]
        Returns: logits value for each class [BATCH_SIZE x SEQ_LENGTH x NUM_CLASSES]
        """
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        transform = self.norm(hidden_states)
        logits = self.mlp(transform)
        return logits

    def _prepare_for_export(self):
        """
        Returns a pair in input, output examples for tracing.
        Returns:
            A pair of (input, output) examples.
        """
        bs = 8
        seq = 64
        input_example = torch.randn(bs, seq, self._hidden_size).to(next(self.parameters()).device)
        output_example = self.forward(hidden_states=input_example)
        return input_example, output_example

    def save_to(self, save_path: str):
        """
        Saves the module to the specified path.
        Args:
            save_path: Path to where to save the module.
        """
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """
        Restores the module from the specified path.
        Args:
            restore_path: Path to restore the module from.
        """
        pass
