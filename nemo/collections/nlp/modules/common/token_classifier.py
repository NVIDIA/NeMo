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

from torch import nn as nn

from nemo.collections.common.parts import MultiLayerPerceptron, transformer_weights_init
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import NeuralType
from nemo.utils.decorators import experimental

__all__ = ['TokenClassifier']

ACT2FN = {"gelu": nn.functional.gelu, "relu": nn.functional.relu}


@experimental
class TokenClassifier(NeuralModule):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    def __init__(
        self,
        hidden_size: object,
        num_classes: object,
        activation: object = 'relu',
        log_softmax: object = True,
        dropout: object = 0.0,
        use_transformer_pretrained: object = True,
    ) -> object:
        super().__init__()
        if activation not in ACT2FN:
            raise ValueError(f'activation "{activation}" not found')
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = ACT2FN[activation]
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.mlp = MultiLayerPerceptron(
            hidden_size, num_classes, num_layers=1, activation=activation, log_softmax=log_softmax
        )
        self.dropout = nn.Dropout(dropout)
        if use_transformer_pretrained:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))

    @typecheck()
    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        transform = self.norm(hidden_states)
        logits = self.mlp(transform)
        return logits

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass
