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

from torch import Tensor

from nemo.collections.common.parts import MultiLayerPerceptron
from nemo.collections.nlp.modules.common.classifier import Classifier
from nemo.core.classes import typecheck
from nemo.core.neural_types import NeuralType, RegressionValuesType

__all__ = ['SequenceRegression']


class SequenceRegression(Classifier):
    """
    Args:
        hidden_size: the hidden size of the mlp head on the top of the encoder
        num_layers: number of the linear layers of the mlp head on the top of the encoder
        activation: type of activations between layers of the mlp head
        dropout: the dropout used for the mlp head
        use_transformer_init: initializes the weights with the same approach used in Transformer
        idx_conditioned_on: index of the token to use as the sequence representation for the classification task,
            default is the first token
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"preds": NeuralType(tuple('B'), RegressionValuesType())}

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        activation: str = 'relu',
        dropout: float = 0.0,
        use_transformer_init: bool = True,
        idx_conditioned_on: int = 0,
    ):
        """ Initializes the SequenceRegression module. """
        super().__init__(hidden_size=hidden_size, dropout=dropout)
        self._idx_conditioned_on = idx_conditioned_on
        self.mlp = MultiLayerPerceptron(
            hidden_size, num_classes=1, num_layers=num_layers, activation=activation, log_softmax=False,
        )
        self.post_init(use_transformer_init=use_transformer_init)

    @typecheck()
    def forward(self, hidden_states: Tensor) -> Tensor:
        """ Forward pass through the module.

        Args:
            hidden_states: hidden states for each token in a sequence, for example, BERT module output
        """
        hidden_states = self.dropout(hidden_states)
        preds = self.mlp(hidden_states[:, self._idx_conditioned_on])
        return preds.view(-1)
