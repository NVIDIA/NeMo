# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pytest
import torch
from nemo.collections.asr.modules.ngpt_decoder import DecoderBlock, NGPTDecoderConfig
from nemo.collections.asr.modules.ngpt_encoder import Block, GPTConfig

@pytest.fixture
def setup_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def setup_shared_weights(setup_device):
    config = GPTConfig(
        n_layers=1,
        n_heads=8,
        n_embd=1024,
        use_nGPT=True,
        bias=False,
    )
    block = Block(config, iblock=0).to(setup_device).to(torch.bfloat16)
    block.eval()
    return block

@pytest.fixture
def setup_decoder_block(setup_device, setup_shared_weights):
    config_decoder = NGPTDecoderConfig(
        vocab_size=8192,
        n_layers=1,
        n_heads=8,
        hidden_size=1024,
        ff_size=3072,
        max_seq_len=1024,
        learn_positional_encodings=False
    )
    decoder_block = DecoderBlock(config_decoder).to(setup_device).to(torch.bfloat16)
    
    # Share weights
    decoder_block.attn.query.weight = setup_shared_weights.query.weight
    decoder_block.attn.key.weight = setup_shared_weights.key.weight
    decoder_block.attn.value.weight = setup_shared_weights.value.weight
    decoder_block.attn.att_c_proj.weight = setup_shared_weights.att_c_proj.weight
    decoder_block.attn.attn_alpha = setup_shared_weights.attn_alpha
    decoder_block.attn.sqk = setup_shared_weights.sqk

    decoder_block.mlp.c_fc.weight = setup_shared_weights.c_fc.weight
    decoder_block.mlp.mlp_c_proj.weight = setup_shared_weights.mlp_c_proj.weight
    decoder_block.mlp.mlp_alpha = setup_shared_weights.mlp_alpha
    decoder_block.mlp.suv = setup_shared_weights.suv

    decoder_block.eval()
    return decoder_block

@pytest.fixture
def setup_block(setup_device, setup_shared_weights):
    block = setup_shared_weights.to(setup_device).to(torch.bfloat16)
    return block

@pytest.fixture
def setup_inputs(setup_device):
    batch_size = 2
    seq_len = 10
    hidden_size = 1024  # This should match the hidden_size in NGPTDecoderConfig
    decoder_query = torch.randn(batch_size, seq_len, hidden_size).to(setup_device).to(torch.bfloat16)
    decoder_mask = torch.ones(batch_size, seq_len).to(setup_device).to(torch.bfloat16)
    decoder_key = torch.randn(batch_size, seq_len, hidden_size).to(setup_device).to(torch.bfloat16)
    encoder_state = torch.randn(batch_size, seq_len, hidden_size).to(setup_device).to(torch.bfloat16)
    encoder_mask = torch.ones(batch_size, seq_len).to(setup_device).to(torch.bfloat16)
    return decoder_query, decoder_mask, decoder_key, encoder_state, encoder_mask

def test_decoder_block_forward(setup_decoder_block, setup_inputs):
    decoder_block = setup_decoder_block
    decoder_query, decoder_mask, decoder_key, encoder_state, encoder_mask = setup_inputs
    with torch.no_grad():
        output = decoder_block(decoder_query, decoder_mask, decoder_key, encoder_state, encoder_mask, cross_attn=False)
    assert output.shape == (decoder_query.size(0), decoder_query.size(1), decoder_query.size(2))

def test_block_forward(setup_block, setup_inputs):
    block = setup_block
    decoder_query, decoder_mask, _, _, _ = setup_inputs
    with torch.no_grad():
        output = block(decoder_query, decoder_mask)
    assert output.shape == (decoder_query.size(0), decoder_query.size(1), decoder_query.size(2))

def test_compare_blocks(setup_decoder_block, setup_block, setup_inputs):
    decoder_block = setup_decoder_block
    block = setup_block
    decoder_query, decoder_mask, decoder_key, encoder_state, encoder_mask = setup_inputs

    output_decoder_block = decoder_block(decoder_query, decoder_mask, decoder_query, encoder_state, encoder_mask, cross_attn=False)
    output_block = block(decoder_query, decoder_mask)

    assert output_decoder_block.shape == output_block.shape
    assert torch.allclose(output_decoder_block, output_block)

if __name__ == '__main__':
    pytest.main()