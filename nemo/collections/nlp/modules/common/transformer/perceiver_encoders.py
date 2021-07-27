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
from nemo.collections.nlp.modules.common.transformer.transformer_modules import AttentionBridge
from nemo.collections.nlp.modules.common.transformer.transformer_decoders import TransformerDecoder
from nemo.collections.nlp.modules.common.transformer.transformer_encoders import TransformerEncoder


__all__ = ["PerceiverEncoder"]


class PerceiverEncoder(TransformerDecoder):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        hidden_steps: int,
        inner_size: int,
        num_attention_heads: int = 1,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        mask_future: bool = False,
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
        # TODO: add to config
        init_hidden_method: str = "att_bridge",
        blocks=2,
    ):
        super().__init__(
            # FIXME: REMOVE ME
            num_layers=1,
            # num_layers=num_layers,
            hidden_size=hidden_size,
            inner_size=inner_size,
            num_attention_heads=num_attention_heads,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            ffn_dropout=ffn_dropout,
            hidden_act=hidden_act,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )

        # FIXME: remove me
        self.blocks = blocks
        # share all weights
        # self.layers = nn.ModuleList([self.layers[0] for _ in range(num_layers)])
        self.final_enc = TransformerEncoder(
            num_layers=3,
            hidden_size=hidden_size,
            inner_size=inner_size,
            mask_future=mask_future,
            num_attention_heads=num_attention_heads,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            ffn_dropout=ffn_dropout,
            hidden_act=hidden_act,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )

        self.init_hidden_method = init_hidden_method

        if self.init_hidden_method == "params":
            # learnable initial hidden values
            self.init_hiddden = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(torch.empty(hidden_steps, hidden_size))
            )
        elif self.init_hidden_method == "att_bridge":
            # initialize latent with attention bridge
            self.att_bridge = AttentionBridge(
                hidden_size=hidden_size,
                k=hidden_steps,
                bridge_size=inner_size,
            )
        else:
            raise ValueError("Unknown init_hidden_method = {init_hidden_method}. Supported methods: params, att_bridge")

        # encoder does not have to be not auto-regressive
        self.diagonal = 0 if mask_future else None

    def forward(self, encoder_states, encoder_mask, hidden_mems_list=None, return_mems=False):
        """
        Args:
            encoder_states: output of the encoder (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc)
            hidden_mems_list: list of the cached hidden states
                for fast autoregressive generation which will be used instead
                of hidden_states as keys and values if not None
            return_mems: bool, whether to return outputs of all encoder layers
                or the last layer only
        """
        if self.init_hidden_method == "params":
            # learnable initial hidden values
            hidden_states = self.init_hiddden
        elif self.init_hidden_method == "att_bridge":
            # initialize latent with attention bridge
            hidden_states = self.att_bridge(
                hidden=encoder_states,
                hidden_mask=encoder_mask,
            )

        # all hidden values are active
        hidden_mask = torch.ones(hidden_states.shape[0], hidden_states.shape[1],
                                 dtype=encoder_mask.dtype, device=encoder_mask.device)

        # FIXME: REMOVE ME
        for block in range(self.blocks):
            hidden_states = super().forward(
                decoder_states=hidden_states,
                decoder_mask=hidden_mask,
                encoder_states=encoder_states,
                encoder_mask=encoder_mask,
                decoder_mems_list=hidden_mems_list,
                return_mems=return_mems,
            )
            hidden_states = self.final_enc(
                encoder_states=hidden_states,
                encoder_mask=hidden_mask,
            )

        return hidden_states

        # return super().forward(
        #     decoder_states=hidden_states,
        #     decoder_mask=hidden_mask,
        #     encoder_states=encoder_states,
        #     encoder_mask=encoder_mask,
        #     decoder_mems_list=hidden_mems_list,
        #     return_mems=return_mems,
        # )
