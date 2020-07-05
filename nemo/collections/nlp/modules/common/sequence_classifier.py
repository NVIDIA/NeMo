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
from nemo.core.neural_types import ChannelType, LogitsType, NeuralType
from nemo.utils.decorators import experimental

__all__ = ['SequenceClassifier']


@experimental
class SequenceClassifier(NeuralModule):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"logits": NeuralType(('B', 'D'), LogitsType())}

    def __init__(
        self,
        hidden_size: object,
        num_classes: object,
        activation: object = 'relu',
        log_softmax: object = True,
        dropout: object = 0.0,
        num_layers: object = 1,
        use_transformer_pretrained: object = True,
    ) -> object:
        super().__init__()
        self.mlp = MultiLayerPerceptron(
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            activation=activation,
            log_softmax=log_softmax,
        )
        self.dropout = nn.Dropout(dropout)
        if use_transformer_pretrained:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))

    @typecheck()
    def forward(self, hidden_states, idx_conditioned_on=0):
        hidden_states = self.dropout(hidden_states)
        logits = self.mlp(hidden_states[:, idx_conditioned_on])
        return logits

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass
