# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned-prefix sliding-window attention: visibility, gradients and cached decoding."""

import copy

import pytest
import torch
from torch.nn import functional as F

from nemo.collections.tts.modules.nemotron_h_decoder import (
    NemotronHAttention,
    NemotronHConfig,
    NemotronHForCausalLM,
    NemotronHModel,
    create_rswa_mask,
    repeat_kv,
)


pytestmark = pytest.mark.unit


def _config(window=3, pattern='*', implementation='sdpa'):
    return NemotronHConfig(
        hidden_size=64,
        num_hidden_layers=len(pattern),
        vocab_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        mamba_num_heads=8,
        mamba_head_dim=8,
        ssm_state_size=16,
        n_groups=2,
        intermediate_size=128,
        hybrid_override_pattern=pattern,
        rswa_window=window,
        _attn_implementation=implementation,
    )


@pytest.fixture(params=['cpu', 'cuda'])
def rswa_device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required for the compiled FlexAttention path')
    return request.param


def _reference_mask(prefix, positions, key_length, window, valid):
    # Independent, small reference; do not use this dense construction in training.
    mask = torch.zeros(len(prefix), 1, len(positions), key_length, dtype=torch.bool)
    for b, p in enumerate(prefix):
        for i, q in enumerate(positions):
            for k in range(key_length):
                mask[b, 0, i, k] = valid[b, q] and valid[b, k] and k <= q and (k < p or k >= q - window + 1)
    return mask


def test_rswa_defaults_and_invalid_configs():
    assert NemotronHConfig().rswa_window is None
    for invalid in (0, -1, True, 2.5):
        with pytest.raises(ValueError, match='positive integer'):
            _config(invalid)
    with pytest.raises(ValueError, match='attention_dropout'):
        NemotronHConfig(rswa_window=3, attention_dropout=0.1)


def test_rswa_exact_visibility_and_padding():
    prefix = torch.tensor([2, 4])
    valid = torch.arange(7)[None] < torch.tensor([7, 5])[:, None]
    mask = create_rswa_mask(prefix, torch.arange(7), 7, 2, valid)
    expected = _reference_mask(prefix.tolist(), list(range(7)), 7, 2, valid)
    torch.testing.assert_close(mask, expected)
    assert mask[0, 0, 6].nonzero().flatten().tolist() == [0, 1, 5, 6]
    assert mask[1, 0, 0].nonzero().flatten().tolist() == [0]  # no future prefix leakage
    assert not mask[1, :, 5:].any()
    self_only = create_rswa_mask(torch.tensor([0]), torch.arange(7), 7, 1)
    torch.testing.assert_close(self_only[0, 0], torch.eye(7, dtype=torch.bool))


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_rswa_attention_matches_dense_outputs_and_gradients(rswa_device, dtype):
    torch.manual_seed(42)
    layer = NemotronHAttention(_config(), 0).to(device=rswa_device, dtype=dtype)
    reference = copy.deepcopy(layer)
    x = torch.randn(2, 137, 64, device=rswa_device, dtype=dtype, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    prefix = torch.tensor([2, 9], device=rswa_device)
    valid = torch.arange(137, device=rswa_device)[None] < torch.tensor([137, 130], device=rswa_device)[:, None]
    mask = create_rswa_mask(prefix, torch.arange(137, device=rswa_device), 137, 3, valid)
    actual = layer(x, attention_mask=mask)[0]
    q = reference.q_proj(ref_x).view(2, 137, 4, 16).transpose(1, 2)
    k = reference.k_proj(ref_x).view(2, 137, 2, 16).transpose(1, 2)
    v = reference.v_proj(ref_x).view(2, 137, 2, 16).transpose(1, 2)
    dense_mask = _reference_mask([2, 9], list(range(137)), 137, 3, valid.cpu()).to(rswa_device)
    expected = F.scaled_dot_product_attention(q, repeat_kv(k, 2), repeat_kv(v, 2), attn_mask=dense_mask)
    expected = reference.o_proj(expected.transpose(1, 2).reshape(2, 137, 64))
    tolerance = 3e-2 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    weights = torch.randn_like(actual)
    (actual.float() * weights).sum().backward()
    (expected.float() * weights).sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=tolerance, atol=tolerance)
    for p, ref_p in zip(layer.parameters(), reference.parameters()):
        if dtype == torch.bfloat16:
            # GQA kernels accumulate shared-KV gradients in a different order.
            # Check relative vector error as well as scale-aware element errors;
            # near-zero entries alone cannot define BF16 relative accuracy.
            error = (p.grad.float() - ref_p.grad.float()).norm()
            assert error <= 0.02 * ref_p.grad.float().norm().clamp_min(1e-8)
            absolute = 0.02 * ref_p.grad.float().abs().max().item()
        else:
            absolute = tolerance
        torch.testing.assert_close(p.grad, ref_p.grad, rtol=tolerance, atol=absolute)


def test_rswa_retains_prefix_and_excludes_distant_and_future_inputs(rswa_device):
    torch.manual_seed(17)
    model = NemotronHModel(_config()).to(rswa_device).eval()
    x = torch.randn(1, 19, 64, device=rswa_device)
    prefix = torch.tensor([2], device=rswa_device)
    with torch.no_grad():
        baseline = model(inputs_embeds=x, context_prefix_lens=prefix).last_hidden_state
        distant = x.clone()
        distant[:, 3] += 5 * torch.randn_like(distant[:, 3])
        changed = model(inputs_embeds=distant, context_prefix_lens=prefix).last_hidden_state
        torch.testing.assert_close(changed[:, -1], baseline[:, -1], rtol=0, atol=0)
        initial = x.clone()
        initial[:, 0] += 5 * torch.randn_like(initial[:, 0])
        changed = model(inputs_embeds=initial, context_prefix_lens=prefix).last_hidden_state
        assert not torch.allclose(changed[:, -1], baseline[:, -1], rtol=1e-6, atol=1e-7)
        future = x.clone()
        future[:, 10:] += 5 * torch.randn_like(future[:, 10:])
        changed = model(inputs_embeds=future, context_prefix_lens=prefix).last_hidden_state
        torch.testing.assert_close(changed[:, :10], baseline[:, :10], rtol=0, atol=0)


@pytest.mark.parametrize('pattern', ['*', 'M*M*'])
@pytest.mark.parametrize('chunks', [[1] * 23, [2, 1, 6, 4, 10]])
def test_rswa_full_sequence_matches_cached_decode_and_prefill(rswa_device, pattern, chunks):
    torch.manual_seed(23)
    model = NemotronHModel(_config(pattern=pattern)).to(rswa_device).eval()
    x = torch.randn(2, 23, 64, device=rswa_device)
    # Prefix can exceed the first prefill chunk; batches have different boundaries.
    prefix = torch.tensor([2, 7], device=rswa_device)
    with torch.no_grad():
        expected = model(inputs_embeds=x, context_prefix_lens=prefix, use_cache=False).last_hidden_state
        cache, start, outputs = None, 0, []
        for length in chunks:
            output = model(
                inputs_embeds=x[:, start : start + length],
                context_prefix_lens=prefix,
                use_cache=True,
                past_key_values=cache,
            )
            cache = output.past_key_values
            outputs.append(output.last_hidden_state)
            start += length
        torch.testing.assert_close(torch.cat(outputs, dim=1), expected, rtol=2e-4, atol=2e-5)


def test_rswa_checkpointing_matches_gradients(rswa_device):
    torch.manual_seed(31)
    model = NemotronHModel(_config(pattern='M*M*')).to(rswa_device).train()
    checked = copy.deepcopy(model)
    checked.gradient_checkpointing = True
    prefix = torch.tensor([2, 5], device=rswa_device)
    x = torch.randn(2, 19, 64, device=rswa_device)
    outputs = []
    for current in (model, checked):
        out = current(inputs_embeds=x, context_prefix_lens=prefix).last_hidden_state
        outputs.append(out)
        out.square().mean().backward()
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    for p, checked_p in zip(model.parameters(), checked.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, checked_p.grad, rtol=1e-5, atol=1e-7)


def test_rswa_requires_explicit_prefix_and_absolute_positions():
    model = NemotronHModel(_config()).eval()
    x = torch.randn(2, 5, 64)
    with pytest.raises(ValueError, match='requires'):
        model(inputs_embeds=x)
    for prefix in (torch.tensor([-1, 2]), torch.tensor([2]), torch.tensor([2.0, 2.0])):
        with pytest.raises(ValueError, match='nonnegative integer'):
            model(inputs_embeds=x, context_prefix_lens=prefix)
    with pytest.raises(ValueError, match='absolute cache timeline'):
        model(inputs_embeds=x, context_prefix_lens=torch.tensor([2, 3]), cache_position=torch.arange(5) + 1)
    with pytest.raises(ValueError, match='2D mask'):
        model(inputs_embeds=x, context_prefix_lens=torch.tensor([2, 3]), attention_mask=torch.ones(2, 4))


def test_rswa_disabled_ignores_prefix_without_new_weights():
    torch.manual_seed(37)
    model = NemotronHModel(_config(None)).eval()
    x = torch.randn(2, 17, 64)
    with torch.no_grad():
        expected = model(inputs_embeds=x).last_hidden_state
        actual = model(inputs_embeds=x, context_prefix_lens=torch.tensor([2, 5])).last_hidden_state
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    enabled = NemotronHModel(_config(3))
    enabled.load_state_dict(model.state_dict(), strict=True)


def test_rswa_wide_window_equals_full_attention_and_flash_selection(rswa_device):
    torch.manual_seed(41)
    model = NemotronHModel(_config(None)).to(rswa_device).eval()
    windowed = NemotronHModel(_config(100, implementation='flash_attention_2')).to(rswa_device).eval()
    windowed.load_state_dict(model.state_dict())
    x = torch.randn(2, 19, 64, device=rswa_device)
    valid = torch.arange(19, device=rswa_device)[None] < torch.tensor([19, 13], device=rswa_device)[:, None]
    with torch.no_grad():
        expected = model(inputs_embeds=x, attention_mask=valid).last_hidden_state
        actual = windowed(
            inputs_embeds=x, attention_mask=valid, context_prefix_lens=torch.tensor([2, 5], device=rswa_device)
        ).last_hidden_state
    # Full attention's historical padding behavior need only match on valid outputs.
    torch.testing.assert_close(actual[valid], expected[valid], rtol=2e-5, atol=2e-5)


def test_causal_lm_routes_prefix_through_generation():
    model = NemotronHForCausalLM(_config()).eval()
    ids = torch.randint(0, 32, (2, 9))
    prefix = torch.tensor([2, 5])
    inputs = model.prepare_inputs_for_generation(ids, context_prefix_lens=prefix)
    assert inputs['context_prefix_lens'] is prefix
    assert model(**inputs).logits.isfinite().all()


def test_rswa_cached_decode_under_inference_mode(rswa_device):
    model = NemotronHModel(_config(pattern='M*')).to(rswa_device).eval()
    with torch.inference_mode():
        # Production streaming creates context tensors inside inference_mode.
        x = torch.randn(2, 11, 64, device=rswa_device)
        prefix = torch.tensor([2, 5], device=rswa_device)
        full = model(inputs_embeds=x, context_prefix_lens=prefix).last_hidden_state
        cache, outputs = None, []
        for i in range(x.size(1)):
            out = model(
                inputs_embeds=x[:, i : i + 1], context_prefix_lens=prefix, use_cache=True, past_key_values=cache
            )
            cache = out.past_key_values
            outputs.append(out.last_hidden_state)
        torch.testing.assert_close(torch.cat(outputs, dim=1), full, rtol=2e-4, atol=2e-5)
