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

from nemo.collections.common.parts import MultiLayerPerceptron
from nemo.collections.nlp.modules.common.classifier import Classifier
from nemo.core.classes import typecheck
from nemo.core.neural_types import LogitsType, NeuralType

__all__ = ['SequenceTokenClassifier']


class SequenceTokenClassifier(Classifier):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "intent_logits": NeuralType(('B', 'D'), LogitsType()),
            "slot_logits": NeuralType(('B', 'T', 'D'), LogitsType()),
        }

    def __init__(
        self,
        hidden_size: int,
        num_intents: int,
        num_slots: int,
        num_layers: int = 2,
        activation: str = 'relu',
        log_softmax: bool = False,
        dropout: float = 0.0,
        use_transformer_init: bool = True,
    ):
        """
        Initializes the SequenceTokenClassifier module, could be used for tasks that train sequence and
        token classifiers jointly, for example, for intent detection and slot tagging task.
        Args:
            hidden_size: hidden size of the mlp head on the top of the encoder
            num_intents: number of the intents to predict
            num_slots: number of the slots to predict
            num_layers: number of the linear layers of the mlp head on the top of the encoder
            activation: type of activations between layers of the mlp head
            log_softmax: applies the log softmax on the output
            dropout: the dropout used for the mlp head
            use_transformer_init: initializes the weights with the same approach used in Transformer
        """
        super().__init__(hidden_size=hidden_size, dropout=dropout)
        self.intent_mlp = MultiLayerPerceptron(
            hidden_size=hidden_size,
            num_classes=num_intents,
            num_layers=num_layers,
            activation=activation,
            log_softmax=log_softmax,
        )
        self.slot_mlp = MultiLayerPerceptron(
            hidden_size=hidden_size,
            num_classes=num_slots,
            num_layers=num_layers,
            activation=activation,
            log_softmax=log_softmax,
        )
        self.post_init(use_transformer_init=use_transformer_init)

    @typecheck()
    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        # intent is classified by first hidden position
        intent_logits = self.intent_mlp(hidden_states[:, 0])
        slot_logits = self.slot_mlp(hidden_states)
        return intent_logits, slot_logits
