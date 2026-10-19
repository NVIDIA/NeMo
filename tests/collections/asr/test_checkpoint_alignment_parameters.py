# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from pathlib import Path

import pytest
import torch

from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.collections.speechlm2.parts import ctc_timestamp_utils as ts


@pytest.fixture
def checkpoint_parts(monkeypatch):
    """Build a tiny calibrated head without any external parameter file."""
    monkeypatch.setattr(ts, '_read_word_alignment_parameters', lambda _: pytest.fail('unexpected YAML read'))
    tokenizer = SentencePieceTokenizer(
        str(Path(__file__).resolve().parents[2] / '.data/asr/tokenizers/an4_spe_128/tokenizer.model')
    )
    config = dict(feat_in=8, num_classes=tokenizer.vocab_size, use_transformer=False, upsampling_factor=2)
    parameters = ts.load_word_alignment_parameters()
    parameters.update(
        speaker_logprob_weight=0.37,
        vad_threshold=0.61,
        boundary_interpolation={'coefficients': [[0.001] * 24, [-0.002] * 24], 'offset': [0.01, -0.02]},
    )
    return ts.TransformerCTCDecoder(**config), tokenizer, config, parameters


@pytest.mark.unit
@pytest.mark.parametrize('suffix,bundled', [('.pt', False), ('.safetensors', False), ('.safetensors', True)])
@pytest.mark.parametrize('override', [None, 0.0])
def test_checkpoint_parameters_roundtrip_and_override(
    tmp_path, monkeypatch, checkpoint_parts, suffix, bundled, override
):
    decoder, tokenizer, config, parameters = checkpoint_parts
    path = ts.save_ctc_timestamp_artifact(
        tmp_path / ('head' + suffix), decoder, tokenizer, config, alignment_parameters=parameters
    )
    if bundled:
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file

        tensors = {'ctc_timestamp.' + k: v for k, v in load_file(path).items()}
        tensors['llm.weight'] = torch.ones(1)
        with safe_open(path, framework='pt') as handle:
            metadata = handle.metadata()
        metadata = {
            ('ctc_timestamp_' + k if k != 'ctc_timestamp_alignment_config' else k): v for k, v in metadata.items()
        }
        path = tmp_path / 'model.safetensors'
        save_file(tensors, path, metadata=metadata)

    restored = ts.load_ctc_timestamp_artifact(path)
    assert restored.alignment_parameters == parameters
    # Embedded settings must remain usable even when the repository YAML is unavailable.
    monkeypatch.setattr(ts, '_read_word_alignment_parameters', lambda _: pytest.fail('read repository YAML'))
    owner = torch.nn.Linear(8, 8)
    owner.ctc_timestamp_alignment = {} if override is None else {'speaker_logprob_weight': override}
    aligner = ts.get_ctc_timestamp_aligner(owner, str(path), torch.device('cpu'))
    assert aligner.speaker_logprob_weight == (0.37 if override is None else override)
    assert aligner.vad_threshold == 0.61
    assert aligner.boundary_interpolation == ts._expand_boundary_calibration(parameters['boundary_interpolation'], 33)
    prepared = aligner.prepare_alignment(
        torch.zeros(1, 32, tokenizer.vocab_size + 1), None, ['hello'], audio_durations=[1.28]
    )
    assert ts.align_prepared_batch(prepared) == aligner.align_prepared(prepared)
    assert ts.get_ctc_timestamp_aligner(owner, str(path), torch.device('cpu')) is aligner
    owner.ctc_timestamp_alignment = {'max_word_frames': 7}
    rebuilt = ts.get_ctc_timestamp_aligner(owner, str(path), torch.device('cpu'))
    assert rebuilt is not aligner and rebuilt.ctc_decoder is aligner.ctc_decoder
    assert rebuilt.max_word_frames == 7 and rebuilt.speaker_logprob_weight == 0.37
    assert rebuilt.boundary_interpolation == aligner.boundary_interpolation


@pytest.mark.unit
@pytest.mark.parametrize('mutation', ['schema', 'basis', 'upsampling', 'unknown', 'missing', 'nonfinite'])
def test_checkpoint_rejects_incompatible_alignment_config(tmp_path, checkpoint_parts, mutation):
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    decoder, tokenizer, config, parameters = checkpoint_parts
    path = ts.save_ctc_timestamp_artifact(
        tmp_path / 'head.safetensors', decoder, tokenizer, config, alignment_parameters=parameters
    )
    with safe_open(path, framework='pt') as handle:
        metadata = handle.metadata()
    embedded = json.loads(metadata['ctc_timestamp_alignment_config'])
    if mutation in ('schema', 'basis', 'upsampling'):
        embedded[
            {'schema': 'schema_version', 'basis': 'basis_version', 'upsampling': 'upsampling_factor'}[mutation]
        ] = 999
    elif mutation == 'unknown':
        embedded['parameters']['not_an_alignment_option'] = 1
    elif mutation == 'missing':
        del embedded['parameters']['vad_threshold']
    else:
        embedded['parameters']['boundary_interpolation']['offset'][0] = float('nan')
    metadata['ctc_timestamp_alignment_config'] = json.dumps(embedded)
    tensors = load_file(path)
    save_file(tensors, tmp_path / 'bad.safetensors', metadata=metadata)
    with pytest.raises(ValueError, match='alignment'):
        ts.load_ctc_timestamp_artifact(tmp_path / 'bad.safetensors')


@pytest.mark.unit
@pytest.mark.parametrize('suffix', ['.pt', '.safetensors'])
def test_legacy_checkpoint_does_not_inherit_another_heads_interpolation(tmp_path, checkpoint_parts, suffix):
    decoder, tokenizer, config, _ = checkpoint_parts
    path = ts.save_ctc_timestamp_artifact(tmp_path / ('legacy' + suffix), decoder, tokenizer, config)
    owner = torch.nn.Linear(8, 8)
    aligner = ts.get_ctc_timestamp_aligner(owner, str(path), torch.device('cpu'))
    assert aligner.boundary_interpolation is None


@pytest.mark.unit
@pytest.mark.parametrize('options,threshold', [({}, 0.61), ({'vad_threshold': 0.5}, 0.5)])
def test_encoder_capture_uses_checkpoint_vad_threshold(tmp_path, checkpoint_parts, options, threshold):
    from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoder

    decoder, tokenizer, config, parameters = checkpoint_parts
    path = ts.save_ctc_timestamp_artifact(
        tmp_path / 'head.safetensors', decoder, tokenizer, config, alignment_parameters=parameters
    )
    owner = torch.nn.Linear(8, 8)
    owner.max_speaker_count, owner.frame_shift_seconds = 1, 0.01
    owner.ctc_timestamp_model_path = str(path)
    owner.ctc_timestamp_alignment = copy.deepcopy(options)
    probs = torch.tensor([[[0.55], [0.65]]])
    inputs = ParallelExpertEncoder._build_ctc_timestamp_inputs(
        owner, torch.zeros(1, 8, 2), torch.tensor([2]), probs, torch.tensor([2]), probs, torch.tensor([2])
    )
    assert torch.equal(inputs.diarization_labels, (probs > threshold).transpose(1, 2))
