# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import torch.nn as nn

from nemo.collections.asr.modules.ngpt_encoder import GPTConfig
from nemo.collections.asr.modules.transformer.transformer_modules import TransformerEmbedding
from nemo.collections.asr.parts.submodules.ngpt_modules import AttentionBlock, MLPBlock


@dataclass
class NGPTDecoderConfig(GPTConfig):
    # Decoder configuration for the nGPT model
    vocab_size: int = 50257
    n_layers: int = 4
    n_heads: int = 8
    hidden_size: int = 1024
    ff_size: int = 3072
    max_seq_len: int = 1024
    learn_positional_encodings: bool = False

class DecoderBlock(nn.Module):
    # Decoder block of the nGPT decoder

    def __init__(self, config):
        super().__init__()

        self.attn = AttentionBlock(config)
        self.cross_attn = AttentionBlock(config)
        self.mlp = MLPBlock(config)

    def forward(self, decoder_query, decoder_mask, decoder_key, encoder_state, encoder_mask):
        # Decoder block
        # order: SA -> Norm -> CA -> Norm -> MLP -> Norm
        h = self.attn(decoder_query, decoder_key, decoder_key, decoder_mask)
        h = self.cross_attn(h, encoder_state, encoder_state, encoder_mask)
        h = self.mlp(h)

        return h

class Decoder(nn.Module):
    def __init__(self, config: NGPTDecoderConfig):
        super().__init__()
        self._config = config

        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
    
    def _get_memory_states(self, decoder_states, decoder_mems_list=None, i=0):
        if decoder_mems_list is not None:
            inp1 = torch.transpose(decoder_mems_list[i], 1, 2)  # Putting seq_len to last dim to handle export cases
            inp2 = torch.transpose(decoder_states, 1, 2)
            memory_states = torch.cat((inp1, inp2), dim=2)
            memory_states = torch.transpose(memory_states, 1, 2)  # Transposing back
        else:
            memory_states = decoder_states
        return memory_states

    def forward(self, decoder_state, decoder_mask, encoder_embeddings, encoder_mask, decoder_mems_list=None, return_mems=False, return_mem_as_list=True):
        # Decoder
        memory_states = self._get_memory_states(decoder_state, decoder_mems_list, 0)
        if return_mems:
            if return_mem_as_list:
                cached_mems_list = [memory_states]
            else:
                cached_mems_list = memory_states.unsqueeze(0)

        for i, decoder_block in enumerate(self.decoder_blocks):
            decoder_state = decoder_block(decoder_state, decoder_mask, decoder_state, encoder_embeddings, encoder_mask)
            memory_states = self._get_memory_states(decoder_state, decoder_mems_list, i+1)
            if return_mems:
                if return_mem_as_list:
                    cached_mems_list.append(memory_states)
                else:
                    cached_mems_list = torch.cat([cached_mems_list, memory_states.unsqueeze(0)], dim=0)
        # import ipdb; ipdb.set_trace()
        if return_mems:
            return cached_mems_list
        return decoder_state

class NGPTDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, n_layers: int, n_heads: int, max_seq_len: int, learn_positional_encodings: bool):
        super().__init__()
        
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._max_seq_len = max_seq_len
        self._learned_pos_enc = learn_positional_encodings

        self._embedding = TransformerEmbedding(
            vocab_size=self._vocab_size,
            hidden_size=self._hidden_size,
            max_sequence_length=self._max_seq_len,
            learn_positional_encodings=self._learned_pos_enc,
        )

        config = NGPTDecoderConfig(
            vocab_size=self._vocab_size,
            hidden_size=self._hidden_size,
            n_layers=self._n_layers,
            n_heads=self._n_heads,
            max_seq_len=self._max_seq_len,
            learn_positional_encodings=self._learned_pos_enc,
            n_embd=self._hidden_size,
            bias=False,
        )


        self._decoder = Decoder(config)

    def forward(self, input_ids, decoder_mask, encoder_embeddings, encoder_mask):
        # Decoder

        start_pos = 0
        decoder_state = self._embedding(input_ids, start_pos=start_pos)

        decoder_state = self._decoder(decoder_state, decoder_mask, encoder_embeddings, encoder_mask)
        return decoder_state

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def max_sequence_length(self):
        return self._max_seq_len

    @property
    def embedding(self):
        return self._embedding

    @property
    def decoder(self):
        return self._decoder