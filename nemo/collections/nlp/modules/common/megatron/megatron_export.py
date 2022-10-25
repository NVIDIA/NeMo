# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import ChannelType, MaskType, NeuralType

__all__ = ["TokensHeadEmb", "DecEmb", "EncEmb"]


class TokensHeadEmb(torch.nn.Module, Exportable):
    """
    Combines decoder_embedding with the tokens_head layer to simulate the classifier in NemoNMT
    """

    def __init__(self, decoder_embedding, tokens_head, device):
        super(TokensHeadEmb, self).__init__()

        self.decoder_embedding = decoder_embedding
        self.tokens_head_bias = tokens_head.bias
        self.device = device

        # properties needed for export
        self.training = False

    def train(self, dummy_input):
        return None

    def modules(self):
        return []

    def forward(self, dec_output):
        if isinstance(dec_output, list):
            dec_output = dec_output[0]

        # dec_output = torch.permute(dec_output, (1, 0, 2))

        if self.tokens_head_bias is not None:
            return F.linear(dec_output, self.decoder_embedding.word_embeddings.weight, self.tokens_head_bias)
        return F.linear(dec_output, self.decoder_embedding.word_embeddings.weight)

    def input_example(self, max_batch=8, max_dim=1024, seq_len=6):
        return [
            torch.randint(low=-3, high=3, size=(max_batch, seq_len, max_dim), device=self.device, dtype=torch.float32)
        ]

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "hidden_states": NeuralType(('B', 'T', 'D'), ChannelType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"log_probs": NeuralType(('B', 'T', 'D'), ChannelType())}

    @property
    def input_names(self) -> List[str]:
        return ['hidden_states']

    @property
    def output_names(self) -> List[str]:
        return ['log_probs']


class DecEmb(torch.nn.Module, Exportable):
    """
    Combines decoder_embedding with the decoder component
    """

    def __init__(self, decoder_embedding, decoder, device):
        super(DecEmb, self).__init__()

        self.decoder_embedding = decoder_embedding
        self.decoder = decoder
        self.device = device

        # properties needed for export
        self.training = False

    def train(self, dummy_input):
        return None

    def modules(self):
        return (self.decoder_embedding, self.decoder)

    def forward(self, input_ids, decoder_mask, encoder_mask, encoder_embeddings, key_mems, val_mems):
        position_ids = (torch.ones((input_ids.size(0), 1)) * (key_mems.size(2) - 1)).long().to(self.device)
        dec_input = self.decoder_embedding(input_ids, position_ids, token_type_ids=None)
        # dec_input, dec_attn_mask, enc_output, enc_attn_mask | dec_input, dec_attn_mask, enc_output, enc_attn_mask
        # _ = dec_mems

        a = self.decoder(
            dec_input, 
            decoder_mask, 
            encoder_embeddings.permute(1, 0, 2),
            encoder_mask, 
            cached_keys=key_mems,
            cached_values=val_mems,
            return_memory=True,
        )
        # .float().permute(2, 0, 1, 3, 4)
        # print(a[0].shape, a[1].permute(2, 0, 1, 3, 4).shape, 'YYYY')
        return a[0].float().permute(1, 0, 2), a[1].float().permute(2, 0, 1, 3, 4)
        # return {"last_hidden_states": a[0].float().permute(1, 0, 2), 
        # "cache": a[1].float().permute(2, 0, 1, 3, 4)}

    def input_example(self, max_batch=8, max_dim=1024, seq_len=6):
        enc_output = torch.randint(
            low=-3, high=3, size=(max_batch, seq_len, max_dim), device=self.device, dtype=torch.float32
        )
        enc_attn_mask = torch.tensor([[1 for _ in range(seq_len)]]).to(self.device)

        dec_len = random.randint(10, 128)
        dec_input = torch.randint(low=0, high=1000, size=(max_batch, 1), device=self.device)
        dec_attn_mask = torch.tensor([[1]]).to(self.device)
        #  decoder_mems = torch.zeros([8, 6, 1024], dtype=torch.float32).to(self.device)
        key_mems = torch.zeros([max_batch, 16, 3, 16, 64], dtype=torch.float32).to(self.device)
        val_mems = torch.zeros([max_batch, 16, 3, 16, 64], dtype=torch.float32).to(self.device)

        # input_ids, decoder_mask, encoder_mask, encoder_embeddings
        return (dec_input, dec_attn_mask, enc_attn_mask, enc_output, key_mems, val_mems)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T', 'D'), ChannelType()),
            "decoder_mask": NeuralType(('B', 'T'), MaskType()),
            "encoder_mask": NeuralType(('B', 'T', 'D'), ChannelType()),
            "encoder_embeddings": NeuralType(('B', 'T'), MaskType()),
            "key_cache": NeuralType(('B', 'D', 'T', 'D', 'D'), ChannelType()),
            "val_cache": NeuralType(('B', 'D', 'T', 'D', 'D'), ChannelType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"last_hidden_states": NeuralType(('B', 'T', 'D'), ChannelType()), 
        "cache": NeuralType(('B', 'D', 'D', 'D', 'D'), ChannelType())}

    @property
    def input_names(self) -> List[str]:
        return ['input_ids', 'decoder_mask', 'encoder_mask', 'encoder_embeddings', 'key_cache', 'val_cache']

    @property
    def output_names(self) -> List[str]:
        return ['last_hidden_states', 'cache']


class EncEmb(torch.nn.Module, Exportable):
    """
    Combines encoder_embedding with the encoder component
    """

    def __init__(self, encoder_embedding, encoder, device):
        super(EncEmb, self).__init__()

        self.encoder_embedding = encoder_embedding
        self.encoder = encoder
        self.device = device

        # properties needed for export
        self.training = False

    def train(self, dummy_input):
        return None

    def modules(self):
        return (self.encoder_embedding, self.encoder)

    def forward(self, input_ids, encoder_mask):
        position_ids = build_position_ids(input_ids)
        enc_input = self.encoder_embedding(input_ids, position_ids, token_type_ids=None)

        # pass input through the encoder
        a = self.encoder(enc_input=enc_input, enc_attn_mask=encoder_mask,).type(torch.float32).permute(1, 0, 2)
        print('ENC:', a.shape)
        return a

    def input_example(self):
        seq_len = random.randint(0, 128)
        return (
            torch.randint(0, 30000, (8, seq_len)).long().to(self.device),
            torch.ones((8, seq_len), dtype=int).to(self.device),
        )

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "encoder_mask": NeuralType(('B', 'T'), MaskType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"last_hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    @property
    def input_names(self) -> List[str]:
        return ['input_ids', 'encoder_mask']

    @property
    def output_names(self) -> List[str]:
        return ['last_hidden_states']
