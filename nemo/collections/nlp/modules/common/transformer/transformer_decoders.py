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

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from omegaconf.omegaconf import MISSING

from nemo.collections.common.parts import form_attention_mask
from nemo.collections.nlp.modules.common.transformer.transformer_modules import MultiHeadAttention, PositionWiseFF
from nemo.core.classes import NeuralModule

__all__ = ["TransformerDecoder"]


class TransformerDecoderBlock(NeuralModule):
    """
    Building block of Transformer decoder.

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
        hidden_size: int,
        inner_size: int,
        num_attention_heads: int = 1,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = False,
    ):
        super().__init__()
        self.pre_ln = pre_ln
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.first_sub_layer = MultiHeadAttention(
            hidden_size, num_attention_heads, attn_score_dropout, attn_layer_dropout
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.second_sub_layer = MultiHeadAttention(
            hidden_size, num_attention_heads, attn_score_dropout, attn_layer_dropout
        )
        self.layer_norm_3 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.third_sub_layer = PositionWiseFF(hidden_size, inner_size, ffn_dropout, hidden_act)

    # TODO: add Neural Types
    def forward(self, decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask):

        # Pre-LN: LN -> Self-Attn -> Drop -> Residual -> LN -> Cross-Attn -> Drop -> Residual -> LN -> FFN
        # Post-LN: Self-Attn -> Drop -> Residual -> LN -> Cross-Attn -> Drop -> Residual -> LN -> FFN -> Residual -> LN
        if self.pre_ln:
            # Share same LN params for query, key (self-attn)
            decoder_query = self.layer_norm_1(decoder_query)
            decoder_keys = self.layer_norm_1(decoder_keys)

        self_attn_output = self.first_sub_layer(decoder_query, decoder_keys, decoder_keys, decoder_mask)
        self_attn_output += decoder_query

        self_attn_output = self.layer_norm_2(self_attn_output) if self.pre_ln else self.layer_norm_1(self_attn_output)

        enc_dec_attn_output = self.second_sub_layer(self_attn_output, encoder_states, encoder_states, encoder_mask)
        enc_dec_attn_output += self_attn_output

        enc_dec_attn_output = (
            self.layer_norm_3(enc_dec_attn_output) if self.pre_ln else self.layer_norm_2(enc_dec_attn_output)
        )

        output_states = self.third_sub_layer(enc_dec_attn_output)

        if not self.pre_ln:
            output_states = self.layer_norm_3(output_states + enc_dec_attn_output)

        return output_states


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        inner_size: int,
        num_attention_heads: int = 1,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = False,
    ):
        super().__init__()

        layer = TransformerDecoderBlock(
            hidden_size,
            inner_size,
            num_attention_heads,
            attn_score_dropout,
            attn_layer_dropout,
            ffn_dropout,
            hidden_act,
            pre_ln,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def _get_memory_states(self, decoder_states, decoder_mems_list=None, i=0):
        if decoder_mems_list is not None:
            memory_states = torch.cat((decoder_mems_list[i], decoder_states), dim=1)
        else:
            memory_states = decoder_states
        return memory_states

    def forward(
        self, decoder_states, decoder_mask, encoder_states, encoder_mask, decoder_mems_list=None, return_mems=False
    ):
        """
        Args:
            decoder_states: output of the embedding layer (B x L_dec x H)
            decoder_mask: decoder inputs mask (B x L_dec)
            encoder_states: output of the encoder (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc)
            decoder_mems_list: list of the cached decoder hidden states
                for fast autoregressive generation which will be used instead
                of decoder_states as keys and values if not None
            return_mems: bool, whether to return outputs of all decoder layers
                or the last layer only
        """
        decoder_attn_mask = form_attention_mask(decoder_mask, diagonal=0)
        encoder_attn_mask = form_attention_mask(encoder_mask)
        memory_states = self._get_memory_states(decoder_states, decoder_mems_list, 0)
        cached_mems_list = [memory_states]

        for i, layer in enumerate(self.layers):
            decoder_states = layer(decoder_states, decoder_attn_mask, memory_states, encoder_states, encoder_attn_mask)
            memory_states = self._get_memory_states(decoder_states, decoder_mems_list, i + 1)
            cached_mems_list.append(memory_states)

        if return_mems:
            return cached_mems_list
        else:
            return cached_mems_list[-1]
