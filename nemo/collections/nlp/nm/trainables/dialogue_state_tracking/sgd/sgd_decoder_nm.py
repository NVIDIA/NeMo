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

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core import ChannelType, EmbeddedTextType, LogitsType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['SGDDecoderNM']


class LogitsAttention(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        """Get logits for elements by using attention on token embedding.
        Args:
          num_classes (int): An int containing the number of classes for which logits are to be generated.
          embedding_dim (int): hidden size of the BERT
    
        Returns:
          A tensor of shape (batch_size, num_elements, num_classes) containing the logits.
        """
        super().__init__()
        self.num_attention_heads = 16
        self.attention_head_size = embedding_dim // self.num_attention_heads
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.dropout = nn.Dropout(0.1)

        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.layer = nn.Linear(embedding_dim, num_classes)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, encoded_utterance, token_embeddings, element_embeddings, utterance_mask):
        """
        token_embeddings: token hidden states from BERT encoding of the utterance
        encoded_utterance: [CLS] token hidden state from BERT encoding of the utterance
        element_embeddings: A tensor of shape (batch_size, num_elements, embedding_dim) extracted from schema
        utterance_mask: binary mask for token_embeddings, 1 for real tokens 0 for padded tokens
        """
        _, num_elements, _ = element_embeddings.size()

        query_layer = self.query(element_embeddings)
        key_layer = self.key(token_embeddings)
        value_layer = self.value(token_embeddings)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if utterance_mask is not None:
            negative_scores = (torch.finfo(attention_scores.dtype).max * -0.7) * torch.ones_like(attention_scores)
            new_x_shape = (utterance_mask.size()[0],) + (1, 1) + (utterance_mask.size()[1],)
            attention_scores = torch.where(
                utterance_mask.view(*new_x_shape).to(bool), attention_scores, negative_scores
            )

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.embedding_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)

        logits = self.layer(context_layer)
        return logits


class Logits(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        """Get logits for elements by conditioning on utterance embedding.
        Args:
          num_classes (int): An int containing the number of classes for which logits are to be generated.
          embedding_dim (int): hidden size of the BERT
    
        Returns:
          A tensor of shape (batch_size, num_elements, num_classes) containing the logits.
        """
        super().__init__()
        self.num_classes = num_classes
        self.utterance_proj = nn.Linear(embedding_dim, embedding_dim)
        self.activation = F.gelu

        self.layer1 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.layer2 = nn.Linear(embedding_dim, num_classes)

    def forward(self, encoded_utterance, token_embeddings, element_embeddings, utterance_mask):
        """
        token_embeddings - token hidden states from BERT encoding of the utterance. Not used
        encoded_utterance - [CLS] token hidden state from BERT encoding of the utterance
        element_embeddings: A tensor of shape (batch_size, num_elements, embedding_dim).
        utterance_mask: binary mask for token_embeddings, 1 for real tokens 0 for padded tokens. Not used
        """
        _, num_elements, _ = element_embeddings.size()

        # Project the utterance embeddings.
        utterance_embedding = self.utterance_proj(encoded_utterance)
        utterance_embedding = self.activation(utterance_embedding)

        # Combine the utterance and element embeddings.
        repeated_utterance_embedding = utterance_embedding.unsqueeze(1).repeat(1, num_elements, 1)

        utterance_element_emb = torch.cat([repeated_utterance_embedding, element_embeddings], axis=2)
        logits = self.layer1(utterance_element_emb)
        logits = self.activation(logits)
        logits = self.layer2(logits)
        return logits


class SGDDecoderNM(TrainableNM):
    """
    Baseline model for schema guided dialogue state tracking with option to make schema embeddings learnable
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module output ports.
        encoded_utterance (float): [CLS] token hidden state from BERT encoding of the utterance
        token_embeddings (float): BERT encoding of utterance (all tokens)
        utterance_mask (bool): Mask which takes the value 0 for padded tokens and 1 otherwise
        cat_slot_values_mask (int): Masks out categorical slots values for slots not used in the service, takes values 0 and 1
        intent_status_mask (int): Masks out padded intents in the service, takes values 0 and 1
        service_ids (int): service ids
        """
        return {
            "encoded_utterance": NeuralType(('B', 'T'), EmbeddedTextType()),
            "token_embeddings": NeuralType(('B', 'T', 'C'), ChannelType()),
            "utterance_mask": NeuralType(('B', 'T'), ChannelType()),
            "cat_slot_values_mask": NeuralType(('B', 'T', 'C'), ChannelType()),
            "intent_status_mask": NeuralType(('B', 'T'), ChannelType()),
            "service_ids": NeuralType(('B'), ChannelType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
            logit_intent_status (float): output for intent status
            logit_req_slot_status (float): output for requested slots status
            logit_cat_slot_status (float): output for categorical slots status
            logit_cat_slot_value (float): output for categorical slots values
            logit_noncat_slot_status (float): Output of SGD model
            logit_noncat_slot_start (float): output for non categorical slots values start
            logit_noncat_slot_end (float): output for non categorical slots values end
        """
        return {
            "logit_intent_status": NeuralType(('B', 'T', 'C'), LogitsType()),
            "logit_req_slot_status": NeuralType(('B', 'T'), LogitsType()),
            "logit_cat_slot_status": NeuralType(('B', 'T', 'C'), LogitsType()),
            "logit_cat_slot_value": NeuralType(('B', 'T', 'C'), LogitsType()),
            "logit_noncat_slot_status": NeuralType(('B', 'T', 'C'), LogitsType()),
            "logit_noncat_slot_start": NeuralType(('B', 'T', 'C'), LogitsType()),
            "logit_noncat_slot_end": NeuralType(('B', 'T', 'C'), LogitsType()),
        }

    def __init__(self, embedding_dim, schema_emb_processor, add_attention_head=False):
        """Get logits for elements by conditioning on utterance embedding.

        Args:
            embedding_dim (int): hidden size of the BERT
            schema_emb_processor (obj): contains schema embeddings for services and config file
            head_transform (str): transformation to use for computing head
        """
        super().__init__()

        # Add a trainable vector for the NONE intent
        self.none_intent_vector = torch.empty((1, 1, embedding_dim), requires_grad=True).to(self._device)
        # TODO truncated norm init
        nn.init.normal_(self.none_intent_vector, std=0.02)
        self.none_intent_vector = torch.nn.Parameter(self.none_intent_vector).to(self._device)

        if add_attention_head:
            projection_module = LogitsAttention
        else:
            projection_module = Logits

        self.intent_layer = Logits(1, embedding_dim).to(self._device)
        self.requested_slots_layer = Logits(1, embedding_dim).to(self._device)

        self.cat_slot_value_layer = projection_module(1, embedding_dim).to(self._device)

        # Slot status values: none, dontcare, active.
        self.cat_slot_status_layer = projection_module(schema_emb_processor._slot_status_size, embedding_dim).to(
            self._device
        )
        self.noncat_slot_layer = projection_module(schema_emb_processor._slot_status_size, embedding_dim).to(
            self._device
        )

        # dim 2 for non_categorical slot - to represent start and end position
        self.noncat_layer1 = nn.Linear(2 * embedding_dim, embedding_dim).to(self._device)
        self.noncat_activation = F.gelu
        self.noncat_layer2 = nn.Linear(embedding_dim, 2).to(self._device)

        config = schema_emb_processor.schema_config
        num_services = len(schema_emb_processor.schemas.services)
        self.intents_emb = nn.Embedding(num_services, config["MAX_NUM_INTENT"] * embedding_dim)
        self.cat_slot_emb = nn.Embedding(num_services, config["MAX_NUM_CAT_SLOT"] * embedding_dim)
        self.cat_slot_value_emb = nn.Embedding(
            num_services, config["MAX_NUM_CAT_SLOT"] * config["MAX_NUM_VALUE_PER_CAT_SLOT"] * embedding_dim
        )
        self.noncat_slot_emb = nn.Embedding(num_services, config["MAX_NUM_NONCAT_SLOT"] * embedding_dim)
        self.req_slot_emb = nn.Embedding(
            num_services, (config["MAX_NUM_CAT_SLOT"] + config["MAX_NUM_NONCAT_SLOT"]) * embedding_dim
        )

        # initialize schema embeddings from the BERT generated embeddings
        schema_embeddings = schema_emb_processor.get_schema_embeddings()
        self.intents_emb.weight.data.copy_(
            torch.from_numpy(np.stack(schema_embeddings['intent_emb']).reshape(num_services, -1))
        )
        self.cat_slot_emb.weight.data.copy_(
            torch.from_numpy(np.stack(schema_embeddings['cat_slot_emb']).reshape(num_services, -1))
        )
        self.cat_slot_value_emb.weight.data.copy_(
            torch.from_numpy(np.stack(schema_embeddings['cat_slot_value_emb']).reshape(num_services, -1))
        )
        self.noncat_slot_emb.weight.data.copy_(
            torch.from_numpy(np.stack(schema_embeddings['noncat_slot_emb']).reshape(num_services, -1))
        )
        self.req_slot_emb.weight.data.copy_(
            torch.from_numpy(np.stack(schema_embeddings['req_slot_emb']).reshape(num_services, -1))
        )

        if not schema_emb_processor.is_trainable:
            self.intents_emb.weight.requires_grad = False
            self.cat_slot_emb.weight.requires_grad = False
            self.cat_slot_value_emb.weight.requires_grad = False
            self.noncat_slot_emb.weight.requires_grad = False
            self.req_slot_emb.weight.requires_grad = False

        self.to(self._device)

    def forward(
        self,
        encoded_utterance,
        token_embeddings,
        utterance_mask,
        cat_slot_values_mask,
        service_ids,
        intent_status_mask,
    ):
        batch_size, emb_dim = encoded_utterance.size()
        intent_embeddings = self.intents_emb(service_ids).view(batch_size, -1, emb_dim)
        cat_slot_emb = self.cat_slot_emb(service_ids).view(batch_size, -1, emb_dim)
        max_number_cat_slots = cat_slot_emb.shape[1]
        cat_slot_value_emb = self.cat_slot_value_emb(service_ids).view(batch_size, max_number_cat_slots, -1, emb_dim)
        noncat_slot_emb = self.noncat_slot_emb(service_ids).view(batch_size, -1, emb_dim)
        req_slot_emb = self.req_slot_emb(service_ids).view(batch_size, -1, emb_dim)

        logit_intent_status = self._get_intents(
            encoded_utterance, intent_embeddings, intent_status_mask, token_embeddings, utterance_mask
        )

        logit_req_slot_status = self._get_requested_slots(
            encoded_utterance, req_slot_emb, token_embeddings, utterance_mask
        )

        logit_cat_slot_status, logit_cat_slot_value = self._get_categorical_slot_goals(
            encoded_utterance, cat_slot_emb, cat_slot_value_emb, cat_slot_values_mask, token_embeddings, utterance_mask
        )

        (
            logit_noncat_slot_status,
            logit_noncat_slot_start,
            logit_noncat_slot_end,
        ) = self._get_noncategorical_slot_goals(encoded_utterance, utterance_mask, noncat_slot_emb, token_embeddings)

        return (
            logit_intent_status,
            logit_req_slot_status,
            logit_cat_slot_status,
            logit_cat_slot_value,
            logit_noncat_slot_status,
            logit_noncat_slot_start,
            logit_noncat_slot_end,
        )

    def _get_intents(self, encoded_utterance, intent_embeddings, intent_status_mask, token_embeddings, utterance_mask):
        """
        Args:
            intent_embedding - BERT schema embeddings
            encoded_utterance - representation of untterance
            intent_status_mask - masks out intent not used for the service
        """
        batch_size = intent_embeddings.size()[0]

        # Add a trainable vector for the NONE intent.
        repeated_none_intent_vector = self.none_intent_vector.repeat(batch_size, 1, 1)
        intent_embeddings = torch.cat([repeated_none_intent_vector, intent_embeddings], axis=1)
        logits = self.intent_layer(
            encoded_utterance=encoded_utterance,
            token_embeddings=token_embeddings,
            element_embeddings=intent_embeddings,
            utterance_mask=utterance_mask,
        )
        logits = logits.squeeze(axis=-1)  # Shape: (batch_size, max_intents + 1)

        # Mask out logits for padded intents
        negative_logits = self._get_negative_logits(logits)
        return torch.where(intent_status_mask.to(dtype=torch.bool), logits, negative_logits)

    def _get_requested_slots(self, encoded_utterance, requested_slot_emb, token_embeddings, utterance_mask):
        """Obtain logits for requested slots."""

        logits = self.requested_slots_layer(
            encoded_utterance=encoded_utterance,
            token_embeddings=token_embeddings,
            element_embeddings=requested_slot_emb,
            utterance_mask=utterance_mask,
        )
        logits = logits.squeeze(axis=-1)

        # logits shape: (batch_size, max_num_slots)
        logits = logits.squeeze(axis=-1)
        return logits

    def _get_categorical_slot_goals(
        self,
        encoded_utterance,
        cat_slot_emb,
        cat_slot_value_emb,
        cat_slot_values_mask,
        token_embeddings,
        utterance_mask,
    ):
        """
        Obtain logits for status and values for categorical slots
        Slot status values: none, dontcare, active
        """

        # Predict the status of all categorical slots.
        status_logits = self.cat_slot_status_layer(
            encoded_utterance=encoded_utterance,
            token_embeddings=token_embeddings,
            element_embeddings=cat_slot_emb,
            utterance_mask=utterance_mask,
        )

        # Predict the goal value.
        # Shape: (batch_size, max_categorical_slots, max_categorical_values, embedding_dim).
        _, max_num_slots, max_num_values, embedding_dim = cat_slot_value_emb.size()
        cat_slot_value_emb_reshaped = cat_slot_value_emb.view(-1, max_num_slots * max_num_values, embedding_dim)

        value_logits = self.cat_slot_value_layer(
            encoded_utterance=encoded_utterance,
            token_embeddings=token_embeddings,
            element_embeddings=cat_slot_value_emb_reshaped,
            utterance_mask=utterance_mask,
        )

        # Reshape to obtain the logits for all slots.
        value_logits = value_logits.view(-1, max_num_slots, max_num_values)

        # Mask out logits for padded slots and values because they will be softmaxed
        negative_value_logits = self._get_negative_logits(value_logits)
        value_logits = torch.where(cat_slot_values_mask.to(dtype=torch.bool), value_logits, negative_value_logits)
        return status_logits, value_logits

    def _get_noncategorical_slot_goals(self, encoded_utterance, utterance_mask, noncat_slot_emb, token_embeddings):
        """
        Obtain logits for status and slot spans for non-categorical slots.
        Slot status values: none, dontcare, active
        """
        # Predict the status of all non-categorical slots.
        max_num_slots = noncat_slot_emb.size()[1]
        status_logits = self.noncat_slot_layer(
            encoded_utterance=encoded_utterance,
            token_embeddings=token_embeddings,
            element_embeddings=noncat_slot_emb,
            utterance_mask=utterance_mask,
        )

        # Predict the distribution for span indices.
        max_num_tokens = token_embeddings.size()[1]

        repeated_token_embeddings = token_embeddings.unsqueeze(1).repeat(1, max_num_slots, 1, 1)
        repeated_slot_embeddings = noncat_slot_emb.unsqueeze(2).repeat(1, 1, max_num_tokens, 1)

        # Shape: (batch_size, max_num_slots, max_num_tokens, 2 * embedding_dim).
        slot_token_embeddings = torch.cat([repeated_slot_embeddings, repeated_token_embeddings], axis=3)

        # Project the combined embeddings to obtain logits, Shape: (batch_size, max_num_slots, max_num_tokens, 2)
        span_logits = self.noncat_layer1(slot_token_embeddings)
        span_logits = self.noncat_activation(span_logits)
        span_logits = self.noncat_layer2(span_logits)

        # Mask out invalid logits for padded tokens.
        utterance_mask = utterance_mask.to(bool)  # Shape: (batch_size, max_num_tokens).
        repeated_utterance_mask = utterance_mask.unsqueeze(1).unsqueeze(3).repeat(1, max_num_slots, 1, 2)
        negative_logits = (torch.finfo(span_logits.dtype).max * -0.7) * torch.ones(
            span_logits.size(), device=self._device, dtype=span_logits.dtype
        )

        span_logits = torch.where(repeated_utterance_mask, span_logits, negative_logits)

        # Shape of both tensors: (batch_size, max_num_slots, max_num_tokens).
        span_start_logits, span_end_logits = torch.unbind(span_logits, dim=3)
        return status_logits, span_start_logits, span_end_logits

    def _get_negative_logits(self, logits):
        # returns tensor with negative logits that will be used to mask out unused values
        # for a particular service
        negative_logits = (torch.finfo(logits.dtype).max * -0.7) * torch.ones(
            logits.size(), device=self._device, dtype=logits.dtype
        )
        return negative_logits
