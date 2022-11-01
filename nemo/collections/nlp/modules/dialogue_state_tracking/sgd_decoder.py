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

import torch
from torch import nn as nn

from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import LogitsType, NeuralType

__all__ = ['SGDDecoder']


class LogitsQA(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int):
        """Get logits for elements by conditioning on input embedding.
        Args:
          num_classes: An int containing the number of classes for which logits are to be generated.
          embedding_dim: hidden size of the BERT
    
        Returns:
          A tensor of shape (batch_size, num_classes) containing the logits.
        """
        super().__init__()
        self.num_classes = num_classes
        self.utterance_proj = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.functional.gelu

        self.layer1 = nn.Linear(embedding_dim, num_classes)

    def forward(self, encoded_utterance):
        """
        Args:
            encoded_utterance: [CLS] token hidden state from BERT encoding of the utterance
        """

        # Project the utterance embeddings.
        utterance_embedding = self.utterance_proj(encoded_utterance)
        utterance_embedding = self.activation(utterance_embedding)

        logits = self.layer1(utterance_embedding)
        return logits


class SGDDecoder(NeuralModule):
    """
    SGDDecoder
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns definitions of module output ports.
        """
        return {
            "logit_intent_status": NeuralType(('B', 'T'), LogitsType()),  #'B'
            "logit_req_slot_status": NeuralType(('B', 'T'), LogitsType()),  #'B'
            "logit_cat_slot_status": NeuralType(('B', 'T'), LogitsType()),
            "logit_cat_slot_value_status": NeuralType(('B', 'T'), LogitsType()),  #'B'
            "logit_noncat_slot_status": NeuralType(('B', 'T'), LogitsType()),
            "logit_spans": NeuralType(('B', 'T', 'D'), LogitsType()),
        }

    def __init__(self, embedding_dim: int) -> None:

        """Get logits for elements by conditioning on utterance embedding.

        Args:
            embedding_dim: hidden size of the BERT
        """
        super().__init__()

        projection_module = LogitsQA

        self.intent_layer = projection_module(1, embedding_dim)
        self.requested_slots_layer = projection_module(1, embedding_dim)

        self.cat_slot_value_layer = projection_module(1, embedding_dim)

        # Slot status values: none, dontcare, active.
        self.slot_status_layer = projection_module(3, embedding_dim)

        # dim 2 for non_categorical slot - to represent start and end position
        self.noncat_layer1 = nn.Linear(embedding_dim, embedding_dim)
        self.noncat_activation = nn.functional.gelu
        self.noncat_layer2 = nn.Linear(embedding_dim, 2)

    @typecheck()
    def forward(self, encoded_utterance, token_embeddings, utterance_mask):
        """
        Args:
            encoded_utterance: [CLS] token hidden state from BERT encoding of the utterance
            token_embeddings: token embeddings from BERT encoding of the utterance
            utterance_mask: utterance mask wiht 0 for padding
        """
        _, _ = encoded_utterance.size()
        logit_intent_status = self._get_intents(encoded_utterance)

        logit_req_slot_status = self._get_requested_slots(encoded_utterance)

        logit_cat_slot_status, logit_cat_slot_value_status = self._get_categorical_slot_goals(encoded_utterance)

        (logit_noncat_slot_status, logit_spans) = self._get_noncategorical_slot_goals(
            encoded_utterance=encoded_utterance, utterance_mask=utterance_mask, token_embeddings=token_embeddings
        )

        return (
            logit_intent_status,
            logit_req_slot_status,
            logit_cat_slot_status,
            logit_cat_slot_value_status,
            logit_noncat_slot_status,
            logit_spans,
        )

    def _get_intents(self, encoded_utterance):
        """Obtain logits for intents.
        Args:
            encoded_utterance: representation of utterance
        """
        logits = self.intent_layer(encoded_utterance=encoded_utterance,)
        return logits

    def _get_requested_slots(self, encoded_utterance):
        """Obtain logits for requested slots.
        Args:
            encoded_utterance: representation of utterance
        """

        logits = self.requested_slots_layer(encoded_utterance=encoded_utterance)
        return logits

    def _get_categorical_slot_goals(self, encoded_utterance):
        """
        Obtain logits for status and values for categorical slots
        Slot status values: none, dontcare, active
        Args:
            encoded_utterance: representation of utterance
        """

        # Predict the status of all categorical slots.
        status_logits = self.slot_status_layer(encoded_utterance=encoded_utterance)

        value_status_logits = self.cat_slot_value_layer(encoded_utterance=encoded_utterance)
        return status_logits, value_status_logits

    def _get_noncategorical_slot_goals(self, encoded_utterance, utterance_mask, token_embeddings):
        """
        Obtain logits for status and slot spans for non-categorical slots.
        Slot status values: none, dontcare, active
        Args:
            encoded_utterance: [CLS] token hidden state from BERT encoding of the utterance
            utterance_mask: utterance mask wiht 0 for padding
            token_embeddings: token embeddings from BERT encoding of the utterance
        """
        status_logits = self.slot_status_layer(encoded_utterance=encoded_utterance)

        # Project the combined embeddings to obtain logits, Shape: (batch_size, max_num_slots, max_num_tokens, 2)
        span_logits = self.noncat_layer1(token_embeddings)
        span_logits = self.noncat_activation(span_logits)
        span_logits = self.noncat_layer2(span_logits)

        # Mask out invalid logits for padded tokens.
        utterance_mask = utterance_mask.to(bool)  # Shape: (batch_size, max_num_tokens).
        repeated_utterance_mask = utterance_mask.unsqueeze(-1)
        negative_logits = (torch.finfo(span_logits.dtype).max * -0.7) * torch.ones(
            span_logits.size(), device=span_logits.get_device(), dtype=span_logits.dtype
        )

        span_logits = torch.where(repeated_utterance_mask, span_logits, negative_logits)

        return status_logits, span_logits

    def _get_negative_logits(self, logits):
        """Returns tensor with negative logits that will be used to mask out unused values for a particular service 
        Args:
            logits: logits whose shape and type will be used to create negative tensor
        """
        negative_logits = (torch.finfo(logits.dtype).max * -0.7) * torch.ones(
            logits.size(), dtype=logits.dtype, device=logits.get_device()
        )
        return negative_logits
