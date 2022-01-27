# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer based language model."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from apex.transformer import parallel_state, tensor_parallel
from apex.transformer.enums import AttnMaskType, LayerType

from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.transformer import ParallelTransformer
from nemo.collections.nlp.modules.common.megatron.utils import (
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
)

__all__ = []

AVAILABLE_ENCODERS = ["MegatronTransformerEncoderDecoderModel"]


class MegatronTransformerEncoderDecoderModel(MegatronModule):
    """Transformer encoder-decoder model.
    """

    def __init__(
        self,
        encoder_input_embedder,
        encoder,
        decoder_input_embedder,
        decoder,
        decoder_output_dist,
    ):
        super(MegatronTransformerEncoderDecoderModel, self).__init__()

        self.encoder_input_embedder = encoder_input_embedder
        self.encoder = encoder
        self.decoder_input_embedder = decoder_input_embedder
        self.decoder = decoder
        self.decoder_output_dist = decoder_output_dist

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""
        self.encoder.set_input_tensor(input_tensor)


    def embed_enc_input(self, enc_emb_input):
        """Embeds encoder observations (e.g., input tokens for text)"""
        return self.encoder_input_embedder(enc_emb_input)

    def encode(self,
        enc_emb_input,
        enc_attn_mask,
        enc_layer_past=None,
        enc_get_key_value=False,
        ):
        """Encodes embedder input using encoder"""
        encoder_input = self.embed_enc_input(enc_emb_input)
        enc_output, enc_output_mask = self.encoder(
            hidden_states=encoder_input,
            attention_mask=enc_attn_mask,
            layer_past=enc_layer_past,
            get_key_value=enc_get_key_value,
        )

        return enc_output, enc_output_mask

    def decode(self,
        enc_output,
        dec_emb_input,
        dec_attn_mask,
        enc_dec_attn_mask
        dec_layer_past=None,
        dec_get_key_value=False,
        ):


    def forward(
        self,
        enc_emb_input,
        enc_attn_mask,
        dec_emb_input,
        dec_attn_mask,
        enc_dec_attn_mask,
        enc_layer_past=None,
        enc_get_key_value=False,
        enc_output=None,
        dec_layer_past=None,
        dec_get_key_value=False,
    ):
        # encoder
        if enc_output is None:
            encoder_output, encoder_output_mask = encode(self,
                enc_emb_input=enc_emb_input,
                enc_attn_mask=enc_attn_mask,
                enc_layer_past=enc_layer_past,
                enc_get_key_value=enc_get_key_value,
            )

        # decoder

        return encoder_output, encoder_output_mask

    # TODO: finish check pointing

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}

        state_dict_[self._model_key] = self.model.state_dict_for_save_checkpoint(destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Encoder.
        if self._model_key in state_dict:
            state_dict_ = state_dict[self._model_key]
        # for backward compatibility.
        elif 'transformer' in state_dict:
            state_dict_ = state_dict['transformer']
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'transformer.' in key:
                    state_dict_[key.split('transformer.')[1]] = state_dict[key]

        # for backward compatibility.
        state_dict_self_attention = {}
        for key in state_dict_.keys():
            if '.attention.' in key:
                state_dict_self_attention[key.replace(".attention.", ".self_attention.")] = state_dict_[key]
            else:
                state_dict_self_attention[key] = state_dict_[key]
        state_dict_ = state_dict_self_attention

        self.model.load_state_dict(state_dict_, strict=strict)
