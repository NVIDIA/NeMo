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
import torch
import random

from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import ChannelType, MaskType, NeuralType
from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from typing import Dict, List, Optional

__all__ = []

class TokensHeadEmb(torch.nn.Module, Exportable):
    """
    Combines decoder_embedding with the tokens_head layer to simulate the classifier in NemoNMT
    """

    def __init__(self, decoder_embedding, tokens_head, device):
        super(TokensHeadEmb, self).__init__()

        self.decoder_embedding = decoder_embedding
        self.tokens_head = tokens_head
        self.device = device

        # properties needed for export
        self.training = False

    def train(self, dummy_input):
        return None

    def modules(self):
        return (
            self.decoder_embedding,
            self.tokens_head,
        )

    def forward(self, dec_output):
        return self.tokens_head(dec_output, self.decoder_embedding.word_embeddings.weight)

    def input_example(self, max_batch=1, max_dim=1024, seq_len=6):
        return torch.randint(
            low=-3, high=3, size=(max_batch, seq_len, max_dim), device=self.device, dtype=torch.float32
        )

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

    def forward(self, input_ids, decoder_mask, encoder_mask, encoder_embeddings, dec_mems):
        position_ids = build_position_ids(input_ids)
        dec_input = self.decoder_embedding(input_ids, position_ids, token_type_ids=None)

        # dec_input, dec_attn_mask, enc_output, enc_attn_mask | dec_input, dec_attn_mask, enc_output, enc_attn_mask
        _ = dec_mems

        return self.decoder(dec_input, decoder_mask, encoder_embeddings, encoder_mask).float()

    def input_example(self, max_batch=1, max_dim=1024, seq_len=6):
        enc_output = torch.randint(
            low=-3, high=3, size=(max_batch, seq_len, max_dim), device=self.device, dtype=torch.float32
        )
        enc_attn_mask = torch.tensor([[1 for _ in range(seq_len)]]).to(self.device)

        dec_len = random.randint(10, 128)
        dec_input = torch.randint(low=0, high=1000, size=(max_batch, dec_len), device=self.device)
        dec_attn_mask = torch.tensor([[1 for _ in range(dec_len)]]).to(self.device)
        decoder_mems = torch.zeros([8, 6, 1024], dtype=torch.float32).to(self.device)

        # input_ids, decoder_mask, encoder_mask, encoder_embeddings
        return (dec_input, dec_attn_mask, enc_attn_mask, enc_output, decoder_mems)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T', 'D'), ChannelType()),
            "decoder_mask": NeuralType(('B', 'T'), MaskType()),
            "encoder_mask": NeuralType(('B', 'T', 'D'), ChannelType()),
            "encoder_embeddings": NeuralType(('B', 'T'), MaskType()),
            "decoder_mems": NeuralType(('B', 'T', 'D'), ChannelType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"last_hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    @property
    def input_names(self) -> List[str]:
        return ['input_ids', 'decoder_mask', 'encoder_mask', 'encoder_embeddings', 'decoder_memes']

    @property
    def output_names(self) -> List[str]:
        return ['last_hidden_states']


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
        return self.encoder(enc_input=enc_input, enc_attn_mask=encoder_mask,).type(torch.float32)

    def input_example(self):
        seq_len = random.randint(0, 128)
        return (
            torch.randint(0, 30000, (1, seq_len)).to(self.device),
            torch.ones((1, seq_len), dtype=int).to(self.device),
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