# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.modules.transformer.transformer import TransformerDecoderNM
from nemo.collections.asr.modules.transformer.transformer_generators import (
    BeamSearchSequenceGenerator,
    BeamSearchSequenceGeneratorWithNGramLM,
    GreedySequenceGenerator,
)
from nemo.collections.asr.parts.submodules.multitask_beam_decoding import TransformerAEDBeamInfer
from nemo.collections.asr.parts.submodules.multitask_greedy_decoding import TransformerAEDGreedyInfer
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.submodules.token_classifier import TokenClassifier


@pytest.fixture()
def deterministic_rng():
    state = torch.get_rng_state()
    torch.manual_seed(0)
    yield
    torch.set_rng_state(state)


@pytest.fixture()
def decoder_nm(deterministic_rng):
    return TransformerDecoderNM(
        vocab_size=8,
        hidden_size=2,
        num_layers=1,
        inner_size=4,
        num_attention_heads=1,
        max_sequence_length=32,
    ).eval()


@pytest.fixture()
def nnet(decoder_nm):
    ans = (
        decoder_nm.embedding,
        decoder_nm.decoder,
        TokenClassifier(hidden_size=2, num_classes=8),
    )
    ans = tuple(m.eval() for m in ans)
    return ans


@pytest.fixture()
def inputs():
    B, T, C = 1, 5, 2
    return (
        torch.tensor([[1]], dtype=torch.long),  # decoder_input_ids
        torch.ones(B, T, C, dtype=torch.float),  # encoder_hidden_states
        torch.ones(B, T, dtype=torch.float),  # encoder_input_mask
    )


@pytest.fixture()
def tokenizer():
    tok = Mock()
    tok.pad = 0
    tok.bos = 1
    tok.eos = 2
    return tok


@pytest.mark.parametrize('with_confidence', [False, True])
def test_greedy_decoding(inputs, nnet, deterministic_rng, with_confidence):
    gen = GreedySequenceGenerator(*nnet, preserve_step_confidence=with_confidence)
    output = gen(*inputs)

    assert len(output) == 3
    best_path, hypotheses, confidence = output

    assert best_path is not None
    assert torch.is_tensor(best_path)
    assert best_path.shape == (1, 25)

    assert hypotheses is None

    if with_confidence:
        assert confidence is not None
        assert torch.is_tensor(confidence)
        assert confidence.shape == best_path.shape
    else:
        assert confidence is None


def test_temperature_sampling_decoding(inputs, nnet):
    gen = GreedySequenceGenerator(*nnet, temperature=10.0, n_samples=2)
    output = gen(*inputs)

    assert len(output) == 3
    best_path, hypotheses, _ = output

    assert best_path is not None
    assert torch.is_tensor(best_path)
    assert best_path.shape[0] == 1

    assert isinstance(hypotheses, list)
    assert len(hypotheses) == 1
    (seq0,) = hypotheses
    assert seq0.shape[0] == 2
    assert (seq0[0] != seq0[1]).any()


def test_beam_decoding_beam_scores_false(inputs, nnet):
    gen = BeamSearchSequenceGenerator(*nnet, beam_size=2)
    output = gen(*inputs, return_beam_scores=False)

    assert len(output) == 1
    (best_path,) = output

    assert best_path is not None
    assert torch.is_tensor(best_path)
    assert best_path.shape == (26,)


def test_beam_decoding_beam_scores_true(inputs, nnet):
    gen = BeamSearchSequenceGenerator(*nnet, beam_size=2)
    output = gen(*inputs, return_beam_scores=True)

    assert len(output) == 3
    beam_paths, scores, best_path = output

    assert beam_paths is not None
    assert isinstance(beam_paths, list)
    assert len(beam_paths) == 1
    (beam_paths_seq0,) = beam_paths
    assert torch.is_tensor(beam_paths_seq0)
    assert beam_paths_seq0.shape == (2, 26)

    assert scores is not None
    assert isinstance(scores, list)
    assert len(scores) == 1
    (scores_seq0,) = scores
    assert torch.is_tensor(scores_seq0)
    assert scores_seq0.shape == (2,)

    assert best_path is not None
    assert torch.is_tensor(best_path)
    assert best_path.shape == (1, 26)


def test_beam_decoding_beam_scores_true_with_lm(inputs, nnet, tmp_path):
    """Test decoding with dummy unigram LM"""
    lm = NGramGPULanguageModel.dummy_unigram_lm(vocab_size=8)
    lm_path = tmp_path / "unigram_lm.nemo"
    lm.save_to(f"{lm_path}")
    gen = BeamSearchSequenceGeneratorWithNGramLM(*nnet, ngram_lm_model=lm_path, ngram_lm_alpha=0.2, beam_size=2)
    output = gen(*inputs, return_beam_scores=True)

    assert len(output) == 3
    beam_paths, scores, best_path = output

    assert beam_paths is not None
    assert isinstance(beam_paths, list)
    assert len(beam_paths) == 1
    (beam_paths_seq0,) = beam_paths
    assert torch.is_tensor(beam_paths_seq0)
    assert beam_paths_seq0.shape == (2, 26)

    assert scores is not None
    assert isinstance(scores, list)
    assert len(scores) == 1
    (scores_seq0,) = scores
    assert torch.is_tensor(scores_seq0)
    assert scores_seq0.shape == (2,)

    assert best_path is not None
    assert torch.is_tensor(best_path)
    assert best_path.shape == (1, 26)


@pytest.fixture()
def prompted_inputs():
    B, T, C = 1, 5, 2
    return (
        torch.tensor([[1, 3, 4, 5, 6]], dtype=torch.long),  # prompt
        torch.ones(B, T, C, dtype=torch.float),  # encoder_hidden_states
        torch.ones(B, T, dtype=torch.float),  # encoder_input_mask
    )


@pytest.fixture()
def batch_prompted_inputs():
    B, T, C = 2, 5, 2
    return (
        torch.tensor([[1, 3, 4, 5, 6], [1, 5, 6, 4, 7]], dtype=torch.long),  # prompt
        torch.ones(B, T, C, dtype=torch.float),  # encoder_hidden_states
        torch.ones(B, T, dtype=torch.float),  # encoder_input_mask
    )


def test_transformer_aed_beam_infer_strips_prompt(prompted_inputs, decoder_nm, nnet, tokenizer):
    decoder_input_ids, encoder_hidden_states, encoder_input_mask = prompted_inputs
    *_, classifier = nnet

    # Run the actual top-level module used by MultiTask AED model for decoding.
    # This module is expected to trim the prompt from the beginning, and eos and pad from the end.
    gen = TransformerAEDBeamInfer(decoder_nm, classifier, tokenizer)
    ans = gen(
        encoder_hidden_states=encoder_hidden_states,
        encoder_input_mask=encoder_input_mask,
        decoder_input_ids=decoder_input_ids,
    )
    best_path = ans[0][0].y_sequence
    assert best_path is not None
    assert torch.is_tensor(best_path)

    # Now run the underlying beam search generator that doesn't trim anything.
    *_, (untrimmed,) = gen.beam_search(*prompted_inputs, return_beam_scores=True)
    assert untrimmed is not None
    assert torch.is_tensor(untrimmed)

    # Check that the expected trimming has indeed been done.
    torch.testing.assert_close(
        untrimmed[decoder_input_ids.shape[1] :], best_path
    )  # stripped the prompt from the beggining


def test_transformer_aed_greedy_infer_strips_prompt(prompted_inputs, decoder_nm, nnet, tokenizer):
    decoder_input_ids, encoder_hidden_states, encoder_input_mask = prompted_inputs
    *_, classifier = nnet

    # Run the actual top-level module used by MultiTask AED model for decoding.
    # This module is expected to trim the prompt from the beginning, and eos and pad from the end.
    gen = TransformerAEDGreedyInfer(decoder_nm, classifier, tokenizer)
    ans = gen(
        encoder_hidden_states=encoder_hidden_states,
        encoder_input_mask=encoder_input_mask,
        decoder_input_ids=decoder_input_ids,
    )
    best_path = ans[0][0].y_sequence
    assert best_path is not None
    assert torch.is_tensor(best_path)

    # Now run the underlying beam search generator that doesn't trim anything.
    (untrimmed,), _, _ = gen.greedy_search(*prompted_inputs)
    assert untrimmed is not None
    assert torch.is_tensor(untrimmed)

    # Check that the expected trimming has indeed been done.
    torch.testing.assert_close(
        untrimmed[decoder_input_ids.shape[1] :], best_path
    )  # stripped the prompt from the beggining


def test_transformer_aed_beam_infer_strips_batch_prompt(batch_prompted_inputs, decoder_nm, nnet, tokenizer):
    """Test batch_size > 1"""
    decoder_input_ids, encoder_hidden_states, encoder_input_mask = batch_prompted_inputs
    *_, classifier = nnet

    # Run the actual top-level module used by MultiTask AED model for decoding.
    # This module is expected to trim the prompt from the beginning, and eos and pad from the end.
    gen = TransformerAEDBeamInfer(decoder_nm, classifier, tokenizer)
    ans = gen(
        encoder_hidden_states=encoder_hidden_states,
        encoder_input_mask=encoder_input_mask,
        decoder_input_ids=decoder_input_ids,
    )
    best_path1 = ans[0][0].y_sequence
    best_path2 = ans[0][1].y_sequence
    assert best_path1 is not None
    assert best_path2 is not None
    assert torch.is_tensor(best_path1)
    assert torch.is_tensor(best_path2)

    # Now run the underlying beam search generator that doesn't trim anything.
    *_, (untrimmed1, untrimmed2) = gen.beam_search(*batch_prompted_inputs, return_beam_scores=True)
    assert untrimmed1 is not None
    assert untrimmed2 is not None
    assert torch.is_tensor(untrimmed1)
    assert torch.is_tensor(untrimmed2)

    # Check that the expected trimming has indeed been done.
    torch.testing.assert_close(
        untrimmed1[decoder_input_ids.shape[1] :], best_path1
    )  # stripped the prompt from the beggining
    torch.testing.assert_close(
        untrimmed2[decoder_input_ids.shape[1] :], best_path2
    )  # stripped the prompt from the beggining


def test_transformer_aed_greedy_infer_strips_batch_prompt(batch_prompted_inputs, decoder_nm, nnet, tokenizer):
    """Test batch_size > 1"""
    decoder_input_ids, encoder_hidden_states, encoder_input_mask = batch_prompted_inputs
    *_, classifier = nnet

    # Run the actual top-level module used by MultiTask AED model for decoding.
    # This module is expected to trim the prompt from the beginning, and eos and pad from the end.
    gen = TransformerAEDGreedyInfer(decoder_nm, classifier, tokenizer)
    ans = gen(
        encoder_hidden_states=encoder_hidden_states,
        encoder_input_mask=encoder_input_mask,
        decoder_input_ids=decoder_input_ids,
    )
    best_path1 = ans[0][0].y_sequence
    best_path2 = ans[0][1].y_sequence
    assert best_path1 is not None
    assert best_path2 is not None
    assert torch.is_tensor(best_path1)
    assert torch.is_tensor(best_path2)

    # Now run the underlying beam search generator that doesn't trim anything.
    (untrimmed1, untrimmed2), _, _ = gen.greedy_search(*batch_prompted_inputs)
    assert untrimmed1 is not None
    assert untrimmed2 is not None
    assert torch.is_tensor(untrimmed1)
    assert torch.is_tensor(untrimmed2)

    # Check that the expected trimming has indeed been done.
    torch.testing.assert_close(
        untrimmed1[decoder_input_ids.shape[1] :], best_path1
    )  # stripped the prompt from the beggining

    torch.testing.assert_close(
        untrimmed2[decoder_input_ids.shape[1] :], best_path2
    )  # stripped the prompt from the beggining


def test_transformer_aed_beam_infer_padded_prompt(prompted_inputs, decoder_nm, nnet, tokenizer):
    """Test that the output of TransformerAEDBeamInfer is the same with and without padding"""
    decoder_input_ids, encoder_hidden_states, encoder_input_mask = prompted_inputs
    # Add padding to the decoder_input_ids
    decoder_input_ids_with_padding = torch.cat([decoder_input_ids, torch.zeros((1, 2), dtype=torch.long)], dim=1)

    *_, classifier = nnet

    # Run the actual top-level module used by MultiTask AED model for decoding.
    # This module is expected to trim the prompt from the beginning, and eos and pad from the end.
    gen = TransformerAEDBeamInfer(decoder_nm, classifier, tokenizer)
    ans = gen(
        encoder_hidden_states=encoder_hidden_states,
        encoder_input_mask=encoder_input_mask,
        decoder_input_ids=decoder_input_ids,
    )
    ans_with_padding = gen(
        encoder_hidden_states=encoder_hidden_states,
        encoder_input_mask=encoder_input_mask,
        decoder_input_ids=decoder_input_ids_with_padding,
    )

    # Extract result tensors
    res = ans[0][0].y_sequence
    res_with_padding = ans_with_padding[0][0].y_sequence

    # Number of output tokens may vary as the number of input tokens is different (5 without padding and 7 with padding)
    # and the randomly initialized model may decode for max_sequence_length steps.
    assert res_with_padding.size(0) <= res.size(
        0
    ), f"Expected len(res_with_padding) <= len(res), got {res_with_padding.size(0)} > {res.size(0)}"
    min_length = res_with_padding.size(0)
    assert torch.equal(
        res[:min_length], res_with_padding
    ), f"Expected ans[:len(ans)] == ans_with_padding, got {res} != {res_with_padding[:res.size(0)]}"


def test_transformer_aed_greedy_infer_padded_prompt(prompted_inputs, decoder_nm, nnet, tokenizer):
    """Test that the output of TransformerAEDGreedyInfer is the same with and without padding"""
    decoder_input_ids, encoder_hidden_states, encoder_input_mask = prompted_inputs
    # Add padding to the decoder_input_ids
    decoder_input_ids_with_padding = torch.cat([decoder_input_ids, torch.zeros((1, 2), dtype=torch.long)], dim=1)

    *_, classifier = nnet

    # Run the actual top-level module used by MultiTask AED model for decoding.
    gen = TransformerAEDGreedyInfer(decoder_nm, classifier, tokenizer)
    ans = gen(
        encoder_hidden_states=encoder_hidden_states,
        encoder_input_mask=encoder_input_mask,
        decoder_input_ids=decoder_input_ids,
    )
    ans_with_padding = gen(
        encoder_hidden_states=encoder_hidden_states,
        encoder_input_mask=encoder_input_mask,
        decoder_input_ids=decoder_input_ids_with_padding,
    )

    # Extract result tensors
    res = ans[0][0].y_sequence
    res_with_padding = ans_with_padding[0][0].y_sequence

    # Number of output tokens may vary as the number of input tokens is different (5 without padding and 7 with padding)
    # because the randomly initialized model may not generate EOS and thus, decode for max_sequence_length steps.
    assert res_with_padding.size(0) <= res.size(
        0
    ), f"Expected len(res_with_padding) <= len(res), got {res_with_padding.size(0)} > {res.size(0)}"
    min_length = res_with_padding.size(0)
    assert torch.equal(
        res[:min_length], res_with_padding
    ), f"Expected ans[:len(ans)] == ans_with_padding, got {res} != {res_with_padding[:res.size(0)]}"
