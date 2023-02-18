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

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn as nn

from nemo.collections.common.parts import MultiLayerPerceptron, SampledMultiLayerPerceptron
from nemo.collections.nlp.modules.common.classifier import Classifier
from nemo.core.classes import typecheck
from nemo.core.neural_types import ChannelType, LabelsType, LogitsType, LogprobsType, NeuralType
from nemo.utils import logging

__all__ = [
    'BertPretrainingTokenClassifier',
    'TokenClassifier',
    'SampledTokenClassifier',
    'SampledTokenClassifierConfig',
    'TokenClassifierConfig',
]

ACT2FN = {"gelu": nn.functional.gelu, "relu": nn.functional.relu}


@dataclass
class TokenClassifierConfig:
    num_layers: int = 1
    activation: str = 'relu'
    log_softmax: bool = True
    dropout: float = 0.0
    use_transformer_init: bool = True


@dataclass
class SampledTokenClassifierConfig(TokenClassifierConfig):
    sampled_softmax: bool = False
    num_samples: Optional[int] = None


class TokenClassifier(Classifier):
    """
    A module to perform token level classification tasks such as Named entity recognition.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns definitions of module output ports.
        """
        if not self.log_softmax:
            return {"logits": NeuralType(('B', 'T', 'C'), LogitsType())}
        else:
            return {"log_probs": NeuralType(('B', 'T', 'C'), LogprobsType())}

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
        super().__init__(hidden_size=hidden_size, dropout=dropout)
        self.log_softmax = log_softmax
        self.mlp = MultiLayerPerceptron(
            hidden_size, num_classes, num_layers=num_layers, activation=activation, log_softmax=log_softmax
        )
        self.post_init(use_transformer_init=use_transformer_init)

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


class SampledTokenClassifier(TokenClassifier):
    """
    A module to perform token level classification tasks such as Named entity recognition.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns definitions of module input ports.
        We implement it here since all NLP classifiers have the same inputs
        """
        return {
            "hidden_states": NeuralType(('B', 'T', 'D'), ChannelType()),
            "targets": NeuralType(('B', 'T'), ChannelType(), optional=True),
            "labels": NeuralType(('B', 'T'), LabelsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns definitions of module output ports.
        """
        if self.training:
            if not self.log_softmax:
                return {
                    "logits": NeuralType(('B', 'T', 'C'), LogitsType()),
                    "targets": NeuralType(('B', 'T'), ChannelType(), optional=True),
                    "labels": NeuralType(('B', 'T'), LabelsType(), optional=True),
                }
            else:
                return {
                    "log_probs": NeuralType(('B', 'T', 'C'), LogprobsType()),
                    "targets": NeuralType(('B', 'T'), ChannelType(), optional=True),
                    "labels": NeuralType(('B', 'T'), LabelsType(), optional=True),
                }
        else:
            return super().output_types

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        activation: str = 'relu',
        log_softmax: bool = True,
        dropout: float = 0.0,
        use_transformer_init: bool = True,
        sampled_softmax: bool = False,
        num_samples: Optional[int] = None,
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
            sampled_softmax: whether to use sampled softmax
            num_samples: minimum number of negative samples to use for sampled softmax
        """
        super().__init__(
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            activation=activation,
            log_softmax=log_softmax,
            dropout=dropout,
            use_transformer_init=use_transformer_init,
        )

        if sampled_softmax:
            self.mlp = SampledMultiLayerPerceptron(
                hidden_size,
                num_classes,
                num_layers=num_layers,
                activation=activation,
                log_softmax=log_softmax,
                num_samples=num_samples,
            )
        else:
            self.mlp = MultiLayerPerceptron(
                hidden_size, num_classes, num_layers=num_layers, activation=activation, log_softmax=log_softmax
            )
        self.sampled_softmax = sampled_softmax
        self.num_samples = num_samples
        self.post_init(use_transformer_init=use_transformer_init)

    @typecheck()
    def forward(self, hidden_states, targets=None, labels=None):
        """
        Performs the forward step of the module.
        Args:
            hidden_states: batch of hidden states (for example, from the BERT encoder module)
                [BATCH_SIZE x SEQ_LENGTH x HIDDEN_SIZE]
        Returns: logits value for each class [BATCH_SIZE x SEQ_LENGTH x NUM_CLASSES]
        """
        hidden_states = self.dropout(hidden_states)

        # If in inference mode, revert to basic token classifier behaviour.
        # Sampled softmax is only used for training.
        if not self.sampled_softmax:
            logits = self.mlp(hidden_states)
            return logits

        # If in eval mode, and sampled softmax is enabled, skip sampled softmax.
        if self.sampled_softmax and (
            self.training is False or torch.is_grad_enabled() is False or torch.is_inference_mode_enabled() is True
        ):
            logits = self.mlp(hidden_states)
            return logits

        # Check if targets are provided for sampled softmax
        if targets is None:
            logging.warning(
                "Sampled Token Classification currently only works with `targets` is provided to the module"
            )
            raise ValueError(
                "Sampled Token Classification only works when the `targets`` are provided during training."
                "Please ensure that you correctly pass the `targets`."
            )

        # If in training mode, and sampled softmax is enabled, use sampled softmax.
        logits, targets, labels = self.mlp(hidden_states, targets, labels)
        return logits, targets, labels


class BertPretrainingTokenClassifier(Classifier):
    """
    A module to perform token level classification tasks for Bert pretraining.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns definitions of module output ports.
        """
        if not self.log_softmax:
            return {"logits": NeuralType(('B', 'T', 'C'), LogitsType())}
        else:
            return {"log_probs": NeuralType(('B', 'T', 'C'), LogprobsType())}

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
        super().__init__(hidden_size=hidden_size, dropout=dropout)

        self.log_softmax = log_softmax

        if activation not in ACT2FN:
            raise ValueError(f'activation "{activation}" not found')
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = ACT2FN[activation]
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.mlp = MultiLayerPerceptron(
            hidden_size, num_classes, num_layers=num_layers, activation=activation, log_softmax=log_softmax
        )
        self.post_init(use_transformer_init=use_transformer_init)

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
