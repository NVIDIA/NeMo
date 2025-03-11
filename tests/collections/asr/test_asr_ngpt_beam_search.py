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

from unittest.mock import Mock

import pytest
import torch

from nemo.collections.asr.modules.ngpt_decoder import NGPTDecoder, NGPTDecoderHead
from nemo.collections.asr.modules.transformer.transformer import TransformerDecoderNM
from nemo.collections.asr.parts.submodules.multitask_beam_decoding import TransformerAEDBeamInfer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@pytest.fixture
def setup_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def setup_tokenizer():
    tokenizer = Mock(spec=TokenizerSpec)
    tokenizer.bos = 0
    tokenizer.eos = 1
    tokenizer.pad = 2
    return tokenizer

@pytest.fixture
def setup_decoder(setup_device):
    decoder = NGPTDecoder(
        vocab_size=8192,
        hidden_size=1024,
        n_layers=2,  # Using 2 layers to test multi-layer behavior
        n_heads=8,
        max_seq_len=1024
    ).to(setup_device).to(torch.bfloat16)
    decoder.eval()
    return decoder

@pytest.fixture
def setup_decoder_head(setup_device):
    head = NGPTDecoderHead(
        hidden_size=1024,
        num_classes=8192,
        num_layers=1,
        log_softmax=True
    ).to(setup_device).to(torch.bfloat16)
    head.eval()
    return head

@pytest.fixture
def setup_beam_search(setup_decoder, setup_decoder_head, setup_tokenizer):
    return TransformerAEDBeamInfer(
        transformer_decoder=setup_decoder,
        log_softmax_module=setup_decoder_head,
        tokenizer=setup_tokenizer,
        beam_size=4,
        length_penalty=0.6,
        max_generation_delta=50
    )

def test_beam_search_basic_functionality(setup_beam_search, setup_device):
    """Test basic beam search functionality with NGPTDecoder"""
    batch_size = 2
    seq_len = 10
    hidden_size = 1024
    
    # Create inputs
    encoder_hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(setup_device).to(torch.bfloat16)
    encoder_input_mask = torch.ones(batch_size, seq_len).to(setup_device).bool()
    
    with torch.no_grad():
        results = setup_beam_search(
            encoder_hidden_states=encoder_hidden_states,
            encoder_input_mask=encoder_input_mask
        )
    
    assert isinstance(results[0], list)
    assert isinstance(results[0][0], Hypothesis)
    assert results[0][0].y_sequence is not None
    assert torch.is_tensor(results[0][0].y_sequence)

def test_beam_search_with_prompt(setup_beam_search, setup_device):
    """Test beam search with prompted decoding using NGPTDecoder"""
    batch_size = 2
    seq_len = 10
    hidden_size = 1024
    prompt_len = 5
    
    # Create inputs with prompt
    encoder_hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(setup_device).to(torch.bfloat16)
    encoder_input_mask = torch.ones(batch_size, seq_len).to(setup_device).bool()
    decoder_input_ids = torch.tensor([[1, 0, 2, 3, 4], [1, 0, 2, 3, 4]], device=setup_device)
    
    with torch.no_grad():
        results = setup_beam_search(
            encoder_hidden_states=encoder_hidden_states,
            encoder_input_mask=encoder_input_mask,
            decoder_input_ids=decoder_input_ids
        )
    
    # Verify prompt handling
    best_path = results[0][0].y_sequence
    assert best_path is not None
    assert len(best_path.shape) == 1, "Expected 1D tensor for best path"
    
    # Run underlying beam search to verify prompt stripping
    with torch.no_grad():
        *_, (untrimmed,) = setup_beam_search.beam_search(
            encoder_hidden_states=encoder_hidden_states,
            encoder_input_mask=encoder_input_mask,
            decoder_input_ids=decoder_input_ids,
            return_beam_scores=True
        )
    
    # Verify prompt stripping
    assert torch.all(untrimmed[:prompt_len] == decoder_input_ids[0])
    torch.testing.assert_close(
        untrimmed[prompt_len:], best_path,
        msg="Beam search failed to correctly strip the prompt"
    )

def test_beam_search_generation_constraints(setup_beam_search, setup_device, setup_tokenizer):
    """Test beam search generation constraints with NGPTDecoder"""
    batch_size = 1
    seq_len = 8
    hidden_size = 1024
    
    # Create inputs
    encoder_hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(setup_device).to(torch.bfloat16)
    encoder_input_mask = torch.ones(batch_size, seq_len).to(setup_device).bool()
    
    with torch.no_grad():
        results = setup_beam_search(
            encoder_hidden_states=encoder_hidden_states,
            encoder_input_mask=encoder_input_mask
        )
    
    best_path = results[0][0].y_sequence
    
    # Verify sequence constraints
    assert not torch.any((best_path[:-1] == setup_tokenizer.eos)), \
        "Found EOS token in the middle of the sequence"
    assert not torch.any((best_path[:-1] == setup_tokenizer.pad)), \
        "Found PAD token in the middle of the sequence"
    
    # Verify sequence length constraints
    max_allowed_length = seq_len + setup_beam_search.beam_search.max_delta_length
    assert best_path.shape[0] <= max_allowed_length, \
        f"Generated sequence length {best_path.shape[0]} exceeds maximum allowed length {max_allowed_length}"



@pytest.fixture()
def deterministic_rng():
    state = torch.get_rng_state()
    torch.manual_seed(0)
    yield
    torch.set_rng_state(state)

def test_transformer_vs_ngpt_decoder(deterministic_rng):
    # Create common parameters for both decoders
    vocab_size = 8
    hidden_size = 4
    num_layers = 2
    num_heads = 2
    max_seq_len = 32

    # Initialize both decoders
    transformer_decoder = TransformerDecoderNM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        inner_size=hidden_size * 4,
        num_attention_heads=num_heads,
        max_sequence_length=max_seq_len,
    ).eval()

    ngpt_decoder = NGPTDecoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layers=num_layers,
        n_heads=num_heads,
        max_seq_len=max_seq_len,
    ).eval()

    # Create test inputs
    batch_size = 2
    seq_len = 5
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    decoder_mask = torch.ones((batch_size, seq_len), dtype=torch.float)
    encoder_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    encoder_mask = torch.ones((batch_size, seq_len), dtype=torch.float)

    # Test without memory states
    transformer_output = transformer_decoder(
        input_ids=input_ids,
        decoder_mask=decoder_mask,
        encoder_embeddings=encoder_hidden_states,
        encoder_mask=encoder_mask,
    )

    ngpt_output = ngpt_decoder(
        input_ids=input_ids,
        decoder_mask=decoder_mask,
        encoder_embeddings=encoder_hidden_states,
        encoder_mask=encoder_mask,
    )

    # Verify outputs have same shape
    assert transformer_output.shape == ngpt_output.shape
    assert transformer_output.shape == (batch_size, seq_len, hidden_size)

    # Test with memory states
    decoder_mems = torch.randn(batch_size, num_layers + 1, seq_len - 1, hidden_size)
    
    transformer_decoder.return_mems = True
    ngpt_decoder.return_mems = True

    transformer_output_with_mems = transformer_decoder(
        input_ids=input_ids[:, -1:],
        decoder_mask=decoder_mask[:, -1:],
        encoder_embeddings=encoder_hidden_states,
        encoder_mask=encoder_mask,
        decoder_mems=decoder_mems,
    )

    ngpt_output_with_mems = ngpt_decoder(
        input_ids=input_ids[:, -1:],
        decoder_mask=decoder_mask[:, -1:],
        encoder_embeddings=encoder_hidden_states,
        encoder_mask=encoder_mask,
        decoder_mems=decoder_mems,
    )

    # Verify memory state outputs have same shape
    assert transformer_output_with_mems.shape == ngpt_output_with_mems.shape
    assert transformer_output_with_mems.shape[0] == num_layers + 1  # Number of layers + 1 for embedding
    assert transformer_output_with_mems.shape[1] == batch_size
    assert transformer_output_with_mems.shape[2] == seq_len
    assert transformer_output_with_mems.shape[3] == hidden_size

    # Test normalization
    transformer_decoder.embedding.normalize_matrices()
    ngpt_decoder.normalize_matrices()

    # Verify both decoders expose same properties
    assert transformer_decoder.hidden_size == ngpt_decoder.hidden_size
    assert transformer_decoder.vocab_size == ngpt_decoder.vocab_size
    assert transformer_decoder.max_sequence_length == ngpt_decoder.max_sequence_length


if __name__ == '__main__':
    pytest.main() 