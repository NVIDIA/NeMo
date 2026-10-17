# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A predicted wait advances the source clock with the training PAD embedding."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.tts.modules.magpietts_inference.phoneme_frontend import TextOnlyPhonemeFrontend


@pytest.mark.unit
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('configured_delay', [0, 3, 8])
def test_frontend_wait_feedback_and_emission(monkeypatch, tmp_path, enabled, configured_delay):
    model = SimpleNamespace(
        cfg=OmegaConf.create({'dynamic_phoneme_delay': enabled, 'use_multiturn_dataset': True}),
        device='cpu',
        phoneme_stacking_factor=1,
        phoneme_tokenizer=SimpleNamespace(bos_token_id=80, eos_token_id=81, pad=0, decode=Mock(return_value='wʌn')),
        tokenizer=SimpleNamespace(encode=Mock(side_effect=[[11] * 5, [12] * 20])),
        text_conditioning_tokenizer_name='text',
        disable_cas_for_context_text=False,
        task_embedding=None,
        repeat_context=False,
        eos_id=13,
        pad_id=0,
        training_modes=[
            SimpleNamespace(
                text_input_mode='streaming',
                streaming_phonemes_delay=configured_delay,
                mode_idx=None,
                name=f'streaming_ph{configured_delay}',
            )
        ],
    )
    forwards = []
    model.embed_text_tokens = lambda ids, **kwargs: torch.ones(1, ids.size(1), 4)
    # A nonzero PAD embedding exposes the training/inference mismatch if not masked.
    model.embed_phoneme_tokens = lambda ids: torch.full((1, 1, 4), 9.0)

    def forward(**kwargs):
        forwards.append(kwargs)
        return SimpleNamespace(last_hidden_state=kwargs['inputs_embeds'], past_key_values='cache')

    ids = iter([0, 7, 81])

    def project(hidden):
        logits = torch.full((1, 1, 82), -100.0)
        logits[..., next(ids)] = 100.0
        return logits

    model.forward, model.phoneme_final_proj = forward, project
    checkpoint = tmp_path / 'fixture.nemo'
    checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(TextOnlyPhonemeFrontend, '_load_model', lambda *args: model)
    decoder = TextOnlyPhonemeFrontend(
        str(checkpoint), 'cpu', expected_delay=99 if enabled else configured_delay, max_steps=3
    )
    result = decoder.generate('$12345', 'en')
    assert result.decoder_steps == 3
    assert result.token_ids == ((7,) if enabled else (0, 7))
    assert result.wait_tokens == int(enabled)
    assert model.phoneme_tokenizer.decode.call_args.args[0] == list(result.token_ids)
    effective_delay = 0 if enabled else configured_delay
    assert decoder.metadata['phoneme_delay'] == effective_delay
    assert forwards[0]['inputs_embeds'].shape[1] == 5 + effective_delay + 1
    assert forwards[1]['cache_position'].item() == 5 + effective_delay + 1
    assert forwards[1]['inputs_embeds'].tolist() == [[[1.0 if enabled else 10.0] * 4]]
