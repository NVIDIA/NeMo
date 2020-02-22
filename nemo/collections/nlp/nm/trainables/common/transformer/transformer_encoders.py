# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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

import copy

import torch
import torch.nn as nn

from nemo.collections.nlp.nm.trainables.common.transformer.transformer_modules import (
    MultiHeadAttention,
    PositionWiseFF,
    TwoStreamSelfAttention,
)
from nemo.collections.nlp.utils.transformer_utils import form_attention_mask

__all__ = []


class TransformerEncoderBlock(nn.Module):
    """
    Building block of Transformer encoder.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of feed-forward
            net, usually is (4-8 x hidden_size) in the papers
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the
            attention layers, but before layer normalization
        ffn_dropout: probability of dropout applied to FFN output
        hidden_act: activation function used between two linear layers in FFN
    """

    def __init__(
        self,
        hidden_size,
        inner_size,
        num_attention_heads=1,
        attn_score_dropout=0,
        attn_layer_dropout=0,
        ffn_dropout=0,
        hidden_act="relu",
    ):
        super().__init__()

        self.first_sub_layer = MultiHeadAttention(
            hidden_size, num_attention_heads, attn_score_dropout, attn_layer_dropout
        )
        self.second_sub_layer = PositionWiseFF(hidden_size, inner_size, ffn_dropout, hidden_act)

    def forward(self, encoder_query, encoder_mask, encoder_keys):
        self_attn_output = self.first_sub_layer(encoder_query, encoder_keys, encoder_keys, encoder_mask)
        output_states = self.second_sub_layer(self_attn_output)
        return output_states


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, mask_future=False, **kwargs):
        super().__init__()

        layer = TransformerEncoderBlock(hidden_size, **kwargs)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.diag = 0 if mask_future else None

    def _get_memory_states(self, encoder_states, encoder_mems_list=None, i=0):
        if encoder_mems_list is not None:
            memory_states = torch.cat((encoder_mems_list[i], encoder_states), dim=1)
        else:
            memory_states = encoder_states
        return memory_states

    def forward(self, encoder_states, encoder_mask, encoder_mems_list=None, return_mems=False):
        """
        Args:
            encoder_states: output of the embedding_layer (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc)
            encoder_mems_list: list of the cached encoder hidden states
                for fast autoregressive generation which will be used instead
                of encoder_states as keys and values if not None
            return_mems: bool, whether to return outputs of all encoder layers
                or the last layer only
        """

        encoder_attn_mask = form_attention_mask(encoder_mask, self.diag)

        memory_states = self._get_memory_states(encoder_states, encoder_mems_list, 0)
        cached_mems_list = [memory_states]

        for i, layer in enumerate(self.layers):
            encoder_states = layer(encoder_states, encoder_attn_mask, memory_states)
            memory_states = self._get_memory_states(encoder_states, encoder_mems_list, i + 1)
            cached_mems_list.append(memory_states)

        if return_mems:
            return cached_mems_list
        else:
            return cached_mems_list[-1]


class XLNetEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        inner_size,
        num_attention_heads=1,
        attn_score_dropout=0,
        attn_layer_dropout=0,
        ffn_dropout=0,
        hidden_act="relu",
    ):
        super().__init__()

        self.first_sub_layer = TwoStreamSelfAttention(
            hidden_size, num_attention_heads, attn_score_dropout, attn_layer_dropout
        )
        self.second_sub_layer = PositionWiseFF(hidden_size, inner_size, ffn_dropout, hidden_act)

    def forward(self, query_states, content_states, query_attn_mask, content_attn_mask):
        output_query_states, output_content_states = self.first_sub_layer(
            query_states, content_states, query_attn_mask, content_attn_mask
        )
        output_content_states = self.second_sub_layer(output_content_states)
        return output_query_states, output_content_states


class XLNetEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, **kwargs):
        super().__init__()

        layer = XLNetEncoderBlock(hidden_size, **kwargs)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, query_states, content_states, input_mask):
        query_attn_mask = form_attention_mask(input_mask, diagonal=-1)
        content_attn_mask = form_attention_mask(input_mask, diagonal=0)
        for layer in self.layers:
            query_states, content_states = layer(query_states, content_states, query_attn_mask, content_attn_mask)
        return query_states, content_states
