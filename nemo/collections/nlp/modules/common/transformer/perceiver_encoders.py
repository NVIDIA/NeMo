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
        inner_size: int,
        num_attention_heads: int = 1,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        mask_future: bool = False,
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
        hidden_steps: int,
        hidden_init_method: str = "params",
        hidden_blocks=2,
    ):
        super().__init__(
            num_layers=1,
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

        self._hidden_steps = hidden_steps
        self._hidden_init_method = hidden_init_method
        self._hidden_blocks = hidden_blocks

        # self-attention encoder
        self.self_enc = TransformerEncoder(
            num_layers=num_layers,
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

        if self._hidden_init_method == "params":
            # learnable initial hidden values
            self.init_hiddden = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(torch.empty(hidden_steps, hidden_size))
            )
        elif self._hidden_init_method == "bridge":
            # initialize latent with attention bridge
            self.att_bridge = AttentionBridge(
                hidden_size=hidden_size,
                k=hidden_steps,
                bridge_size=inner_size,
            )
        else:
            raise ValueError("Unknown hidden_init_method = {hidden_init_method}. Supported methods: params, att_bridge")

        # encoder does not have to be not auto-regressive
        self.diagonal = 0 if mask_future else None


    @property
    def hidden_steps(self):
        return self._hidden_steps

    @property
    def hidden_blocks(self):
        return self._hidden_blocks

    @property
    def hidden_init_method(self):
        return self._hidden_init_method

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
        # all hidden values are active
        hidden_mask = torch.ones(encoder_states.shape[0], self._hidden_steps,
                                 dtype=encoder_mask.dtype, device=encoder_mask.device)

        if self._hidden_init_method == "params":
            # initialize latent with learned parameters
            hidden_states = self.init_hiddden
        elif self._hidden_init_method == "att_bridge":
            # initialize latent with attention bridge
            hidden_states = self.att_bridge(
                hidden=encoder_states,
                hidden_mask=encoder_mask,
            )

        # apply block (cross-attention, self-attention) multiple times
        for block in range(self._hidden_blocks):
            # cross attention of hidden over input
            hidden_states = super().forward(
                decoder_states=hidden_states,
                decoder_mask=hidden_mask,
                encoder_states=encoder_states,
                encoder_mask=encoder_mask,
                decoder_mems_list=hidden_mems_list,
                return_mems=return_mems,
            )

            # self-attention over hidden
            hidden_states = self.self_enc(
                encoder_states=hidden_states,
                encoder_mask=hidden_mask,
            )

        return hidden_states, hidden_mask