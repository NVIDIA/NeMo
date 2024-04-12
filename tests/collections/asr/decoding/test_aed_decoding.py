import pytest
import torch

from nemo.collections.asr.modules.transformer import TransformerDecoder, TransformerEmbedding
from nemo.collections.asr.modules.transformer.transformer_generators import (
    BeamSearchSequenceGenerator,
    GreedySequenceGenerator,
)
from nemo.collections.asr.parts.submodules.token_classifier import TokenClassifier


@pytest.fixture()
def deterministic_rng():
    state = torch.get_rng_state()
    torch.manual_seed(0)
    yield
    torch.set_rng_state(state)


@pytest.fixture()
def nnet(deterministic_rng):
    ans = (
        TransformerEmbedding(vocab_size=8, hidden_size=2, max_sequence_length=32),
        TransformerDecoder(num_layers=1, hidden_size=2, inner_size=4),
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


def test_greedy_decoding(inputs, nnet):
    gen = GreedySequenceGenerator(*nnet)
    output = gen(*inputs)

    assert len(output) == 2
    best_path, hypotheses = output

    assert best_path is not None
    assert torch.is_tensor(best_path)
    assert best_path.shape == (1, 25)

    assert hypotheses is None


def test_temperature_sampling_decoding(inputs, nnet):
    gen = GreedySequenceGenerator(*nnet, temperature=10.0, n_samples=2)
    output = gen(*inputs)

    assert len(output) == 2
    best_path, hypotheses = output

    assert best_path is not None
    assert torch.is_tensor(best_path)
    assert best_path.shape == (1, 25)

    assert isinstance(hypotheses, list)
    assert len(hypotheses) == 1
    (seq0,) = hypotheses
    assert seq0.shape == (2, 25)


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
