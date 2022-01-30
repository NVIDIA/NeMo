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
    build_position_ids,
    enc_dec_extended_attention_mask
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
    ):
        super(MegatronTransformerEncoderDecoderModel, self).__init__()

        self.encoder_input_embedder = encoder_input_embedder
        self.encoder = encoder
        self.decoder_input_embedder = decoder_input_embedder
        self.decoder = decoder

        self._encoder_input_embedder_key = "encoder_input_embedder"
        self._encoder_key = "encoder"
        self._decoder_input_embedder_key = "decoder_input_embedder"
        self._decoder_key = "decoder"


    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""
        self.encoder.set_input_tensor(input_tensor)

    def embed_enc_input(self, enc_emb_input):
        """Embeds encoder input (e.g., input tokens for text)"""
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

    def embed_dec_input(self, dec_emb_input):
        """Embeds decoder input (e.g., input tokens for text)"""
        return self.decoder_input_embedder(dec_emb_input)

    def decode(self,
               enc_output,
               dec_emb_input,
               dec_attn_mask,
               enc_dec_attn_mask
               dec_layer_past=None,
               dec_get_key_value=False,
               ):
        """Decodes embedder input using decoder and encoder input"""

        dec_input = self.embed_dec_input(dec_emb_input)
        dec_output = self.decoder(
            dec_input,
            dec_attn_mask,
            layer_past=dec_layer_past,
            get_key_value=dec_get_key_value,
            encoder_output=enc_output,
            enc_dec_attn_mask=enc_dec_attn_mask,
        )

        return dec_output

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
            enc_output, enc_output_mask = self.encode(
                enc_emb_input=enc_emb_input,
                enc_attn_mask=enc_attn_mask,
                enc_layer_past=enc_layer_past,
                enc_get_key_value=enc_get_key_value,
            )

        # decoder
        dec_output = self.decode(
            enc_output,
            dec_emb_input,
            dec_attn_mask,
            enc_dec_attn_mask
            dec_layer_past=None,
            dec_get_key_value=False,
        )

        return enc_output, enc_output_mask, dec_output


    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}

        state_dict_[self._encoder_input_embedder_key] = self.encoder_input_embedder.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        state_dict_[self._encoder_key] = self.encoder.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        state_dict_[self._decoder_input_embedder_key] = self.decoder_input_embedder.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        state_dict_[self._decoder_key] = self.decoder.state_dict_for_save_checkpoint(destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""


        self.encoder_input_embedder.encoder_input_embedderload_state_dict(state_dict[self._encoder_input_embedder_key], strict=strict)
        self.encoder.load_state_dict(state_dict[self._encoder_key], strict=strict)
        self.decoder_input_embedder.load_state_dict(state_dict[self._decoder_input_embedder_key], strict=strict)
        self.decoder.load_state_dict(state_dict[self._decoder_key], strict=strict)
