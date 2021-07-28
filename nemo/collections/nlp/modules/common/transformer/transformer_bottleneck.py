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
from typing import Optional

import torch
from omegaconf.omegaconf import MISSING

from nemo.collections.nlp.modules.common.transformer.transformer import (
    TransformerEncoderNM,
    TransformerDecoderNM,
)
from nemo.collections.nlp.modules.common.decoder_module import DecoderModule
from nemo.collections.nlp.modules.common.encoder_module import EncoderModule
from nemo.collections.nlp.modules.common.transformer.transformer_decoders import TransformerDecoder
from nemo.collections.nlp.modules.common.transformer.transformer_encoders import TransformerEncoder
from nemo.collections.nlp.modules.common.transformer.transformer import NeMoTransformerConfig
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable


@dataclass
class NeMoTransformerBottleneckConfig(NeMoTransformerConfig):
    # architecture details (default is no bottleneck)
    arch: str = ''
    hidden_steps: int = -1


@dataclass
class NeMoTransformerBottleneckEncoderConfig(NeMoTransformerBottleneckConfig):
    mask_future: bool = False


@dataclass
class NeMoTransformerBottleneckDecoderConfig(NeMoTransformerBottleneckConfig):
    r2l: bool = False


class TransformerBottleneckEncoderNM(TransformerEncoderNM):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        inner_size: int,
        num_attention_heads: int,
        max_sequence_length: int = 512,
        num_token_types: int = 2,
        embedding_dropout: float = 0.0,
        learn_positional_encodings: bool = False,
        ffn_dropout: float = 0.0,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        hidden_act: str = 'relu',
        mask_future: bool = False,
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
        arch='',
        hidden_steps=-1,
    ):
        super().__init__()

        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._max_sequence_length = max_sequence_length

        self._embedding = TransformerEmbedding(
            vocab_size=self._vocab_size,
            hidden_size=self._hidden_size,
            max_sequence_length=max_sequence_length,
            num_token_types=num_token_types,
            embedding_dropout=embedding_dropout,
            learn_positional_encodings=learn_positional_encodings,
        )

        self._encoder = TransformerEncoder(
            hidden_size=self._hidden_size,
            num_layers=num_layers,
            inner_size=inner_size,
            num_attention_heads=num_attention_heads,
            ffn_dropout=ffn_dropout,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            hidden_act=hidden_act,
            mask_future=mask_future,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )

    @typecheck()
    def forward(self, input_ids, encoder_mask):
        embeddings = self._embedding(input_ids=input_ids)
        encoder_hidden_states = self._encoder(encoder_states=embeddings, encoder_mask=encoder_mask)
        return encoder_hidden_states

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def max_sequence_length(self):
        return self._max_sequence_length

    @property
    def embedding(self):
        return self._embedding

    @property
    def encoder(self):
        return self._encoder

    def input_example(self):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        sample = next(self.parameters())
        input_ids = torch.randint(low=0, high=2048, size=(2, 16), device=sample.device)
        encoder_mask = torch.randint(low=0, high=1, size=(2, 16), device=sample.device)
        return tuple([input_ids, encoder_mask])


class TransformerBottleneckDecoderNM(TransformerDecoderNM):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        inner_size: int,
        num_attention_heads: int,
        max_sequence_length: int = 512,
        num_token_types: int = 2,
        embedding_dropout: float = 0.0,
        learn_positional_encodings: bool = False,
        ffn_dropout: float = 0.0,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        hidden_act: str = 'relu',
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
        arch='',
        hidden_steps=-1,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            inner_size=inner_size,
            num_attention_heads=num_attention_heads,
            max_sequence_length=max_sequence_length,
            num_token_types=num_token_types,
            embedding_dropout=embedding_dropout,
            learn_positional_encodings=learn_positional_encodings,
            ffn_dropout=ffn_dropout,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            hidden_act=hidden_act,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )

        self._arch = arch
        self._hidden_steps = hidden_steps

        # replace decoder
        self._decoder = self._build_decoder(
            arch=arch,
            hidden_steps=hidden_steps,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            inner_size=inner_size,
            num_attention_heads=num_attention_heads,
            ffn_dropout=ffn_dropout,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            hidden_act=hidden_act,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
            )

    def _build_decoder(self, arch, **kwargs):
        """
        Returns a decoder based on architecture arch and kwargs
        """
        if arch not in self.supported_arch:
            raise ValueError("Unknown arch = {arch}, supported arch = {supported arch}".format(
                arch=arch,
                supported_arch=self.supported_arch,
            ))

        # usual non-bottleneck transformer decoder
        if (not arch) or (arch == "seq2seq"):
            decoder = TransformerDecoder(**kwargs)

        return decoder

    @property
    def supported_arch(self):
        return [None, "", "seq2seq"]

    @property
    def arch(self):
        return self._arch

    @property
    def hidden_steps(self):
        return self._hidden_steps
