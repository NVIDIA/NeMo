# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched cached decoding preserves each sample's prefix, clock, waits and EOS."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.modules.magpietts_inference.phoneme_frontend import (
    generate_batched_text_only_phoneme_tokens,
    generate_text_only_phoneme_tokens,
)
from nemo.collections.tts.modules.nemotron_h_decoder import NemotronHConfig, NemotronHModel
from tests.collections.tts.test_phoneme_delay_metrics import batch

pytestmark = pytest.mark.unit


class BatchDecoder(torch.nn.Module):
    def __init__(self, scripts, dynamic=True, repeat=False, task=False):
        super().__init__()
        self.cfg = OmegaConf.create({'use_multiturn_dataset': True, 'dynamic_phoneme_delay': dynamic})
        self.phoneme_tokenizer = SimpleNamespace(pad=0, bos_token_id=80, eos_token_id=81)
        self.phoneme_stacking_factor = 1
        self.disable_cas_for_context_text = False
        self.repeat_context = repeat
        self.pad_id, self.eos_id = 0, 13
        self.task_embedding = torch.nn.Embedding(1, 4) if task else None
        self.mode = SimpleNamespace(
            text_input_mode='streaming', streaming_phonemes_delay=8, mode_idx=0 if task else None
        )
        self.scripts = scripts
        self.forwards = []
        self.logits = []
        self.eval()

    def embed_text_tokens(self, ids, **kwargs):
        return ids[:, :, None].float().expand(-1, -1, 4).clone()

    def embed_phoneme_tokens(self, ids):
        return ids[:, 0, :, None].float().expand(-1, -1, 4).clone()

    def _repeat_gated_context_embeddings(self, context, lengths, positions):
        return context[:, positions % context.size(1)] * 0.5

    def forward(self, **kwargs):
        self.forwards.append(kwargs)
        if kwargs.get('past_key_values') is None:
            # Context IDs encode sample identities. Read them before task/prefix
            # repetition changes the combined model input.
            context = self.active_context
            ids = context[:, 0].long()
            step = 0
        else:
            ids, step = kwargs['past_key_values']
            step += 1
        hidden = kwargs['inputs_embeds'].new_zeros(ids.numel(), 1, 4)
        hidden[:, 0, 0], hidden[:, 0, 1] = ids, step
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=(ids, step))

    def phoneme_final_proj(self, hidden):
        result = hidden.new_full((hidden.size(0), 1, 82), -100)
        for i, (sample, step) in enumerate(hidden[:, 0, :2].long().tolist()):
            script = self.scripts[sample]
            result[i, 0, script[min(step, len(script) - 1)]] = 100
        return result


def decoder(scripts, **kwargs):
    model = BatchDecoder(scripts, **kwargs)
    embed = model.embed_text_tokens

    def with_context(ids, **options):
        if 'disable_cas_embedding' in options:
            model.active_context = ids
        return embed(ids, **options)

    model.embed_text_tokens = with_context
    return model


def inputs():
    return (
        torch.tensor([[11, 13, 99, 99], [11, 12, 13, 99], [11, 12, 12, 13], [11, 13, 99, 99]]),
        torch.tensor([2, 3, 4, 2]),
        torch.tensor([[21, 22, 99], [23, 24, 25], [26, 27, 99], [28, 29, 30]]),
        torch.tensor([2, 3, 2, 3]),
    )


@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('repeat,task', [(False, False), (True, True)])
def test_batch_matches_serial_inputs_and_outputs_with_variable_prefixes(dynamic, repeat, task):
    scripts = {21: [7, 81], 23: [0, 8, 0, 81], 26: [81], 28: [9]}
    model = decoder(scripts, dynamic=dynamic, repeat=repeat, task=task)
    text, text_lens, context, context_lens = inputs()
    actual = generate_batched_text_only_phoneme_tokens(model, text, text_lens, context, context_lens, model.mode, 5)
    # Two prefix cohorts: 2+5 forward calls instead of 2+4+1+5 serial calls.
    assert len(model.forwards) == 7
    assert all(call['inputs_embeds'].size(0) == 2 for call in model.forwards)
    assert [value.predicted_tokens for value in actual] == [(7, 81), (0, 8, 0, 81), (81,), (9,) * 5]
    assert [value.decoder_steps for value in actual] == [2, 4, 1, 5]
    assert [value.reached_eos for value in actual] == [True, True, True, False]
    assert actual[2].first_phoneme_step is None
    for i in range(4):
        serial = decoder(scripts, dynamic=dynamic, repeat=repeat, task=task)
        serial.load_state_dict(model.state_dict())
        expected = generate_text_only_phoneme_tokens(
            serial, text[i : i + 1, : text_lens[i]], context[i : i + 1, : context_lens[i]], serial.mode, 5
        )
        assert actual[i] == expected
        # Invalid source padding (99) must never become a decoder input.
        cohort_calls = [call for call in model.forwards if call['context_prefix_lens'][0] == context_lens[i] + task]
        row = 0 if i < 2 else 1
        for batch_call, serial_call in zip(cohort_calls, serial.forwards):
            torch.testing.assert_close(batch_call['inputs_embeds'][row : row + 1], serial_call['inputs_embeds'])
            torch.testing.assert_close(batch_call['cache_position'], serial_call['cache_position'])


def test_adaptive_safety_caps_are_per_sample_not_batch_maximum():
    model = decoder({21: [0], 23: [7]})
    text = torch.full((2, 20), 11)
    result = generate_batched_text_only_phoneme_tokens(
        model, text, torch.tensor([2, 20]), torch.tensor([[21], [23]]), torch.tensor([1, 1]), model.mode, None
    )
    assert [row.decoder_steps for row in result] == [256, 288]
    assert result[0].wait_tokens == 256 and result[0].first_phoneme_step is None
    assert len(result[1].token_ids) == 288
    assert not any(row.reached_eos for row in result)
    assert len(model.forwards) == 288


def test_validation_runs_one_batched_loop_and_honors_remaining_sample_budget():
    model = decoder({21: [0, 7, 81]})
    model.training_modes = [model.mode]
    model.cfg.text_only_val_ar_max_samples = 3
    model.cfg.text_only_val_ar_max_steps = 5
    model._text_only_val_ar_samples = 1
    model.trainer = SimpleNamespace(sanity_checking=False)
    data = {key: value.repeat((4,) + (1,) * (value.ndim - 1)) for key, value in batch().items()}
    result = EasyMagpieTTSModel._text_only_validation_delay_stats(model, data)
    assert result['val_text_only_target_delay_counts'][0] == 4
    assert result['val_text_only_ar_delay_counts'][0] == 2
    assert model._text_only_val_ar_samples == 3
    assert len(model.forwards) == 3
    assert all(call['inputs_embeds'].size(0) == 2 for call in model.forwards)


class HybridDecoder(BatchDecoder):
    def __init__(self, window):
        super().__init__({}, repeat=True, task=True)
        config = NemotronHConfig(
            hidden_size=32,
            num_hidden_layers=3,
            vocab_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            mamba_num_heads=4,
            mamba_head_dim=8,
            ssm_state_size=8,
            n_groups=2,
            intermediate_size=64,
            hybrid_override_pattern='M*-',
            rswa_window=window,
            _attn_implementation='sdpa',
        )
        self.backbone = NemotronHModel(config).eval()
        self.text_embedding = torch.nn.Embedding(128, 32)
        self.phoneme_embedding = torch.nn.Embedding(82, 32)
        self.task_embedding = torch.nn.Embedding(1, 32)
        self.projection = torch.nn.Linear(32, 82)

    def embed_text_tokens(self, ids, **kwargs):
        return self.text_embedding(ids)

    def embed_phoneme_tokens(self, ids):
        return self.phoneme_embedding(ids[:, 0])

    def forward(self, **kwargs):
        return self.backbone(**kwargs)

    def phoneme_final_proj(self, hidden):
        logits = self.projection(hidden)
        self.logits.append(logits.clone())
        return logits


@pytest.mark.parametrize('window', [None, 3])
def test_actual_hybrid_cache_logits_match_serial_for_live_samples(window):
    torch.manual_seed(137)
    model = HybridDecoder(window).eval()
    text, lengths, context, context_lens = inputs()
    # Same-length prefixes share one hybrid cache, while text lengths differ.
    context, context_lens = context[[0, 2]], context_lens[[0, 2]]
    text, lengths = text[[0, 2]], lengths[[0, 2]]
    actual = generate_batched_text_only_phoneme_tokens(model, text, lengths, context, context_lens, model.mode, 6)
    batched_logits = list(model.logits)
    for i in range(2):
        model.logits.clear()
        expected = generate_text_only_phoneme_tokens(
            model, text[i : i + 1, : lengths[i]], context[i : i + 1, : context_lens[i]], model.mode, 6
        )
        assert actual[i] == expected
        for batch_logits, serial_logits in zip(batched_logits, model.logits):
            torch.testing.assert_close(batch_logits[i : i + 1], serial_logits, rtol=2e-4, atol=2e-5)
