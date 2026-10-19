# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-weighted first-phoneme timing and token-weighted wait coverage."""

from types import SimpleNamespace

import torch


TEXT_ONLY_EXAMPLE_COLUMNS = [
    'global_step',
    'epoch',
    'rank',
    'sample_id',
    'dataset',
    'language',
    'text_tokenizer',
    'input_text',
    'raw_text',
    'reference_ipa',
    'generated_ipa',
    'exact_phoneme_match',
    'target_first_phoneme_step',
    'generated_first_phoneme_step',
    'target_wait_percent',
    'generated_wait_percent',
    'generated_steps',
    'generated_wait_tokens',
    'reached_eos',
    'truncated',
    'no_phoneme',
    'prediction_mode',
]


def phoneme_decoding_example(reference_tokens, generated, tokenizer, *, fixed_delay=0, metadata=None):
    """Make a CPU-only W&B row from an existing decode, without running the model."""
    reference = reference_tokens.detach().cpu().tolist()
    special = {tokenizer.pad, tokenizer.bos_token_id, tokenizer.eos_token_id}
    reference_content = [token for token in reference if token not in special]
    predicted_content = [token for token in generated.predicted_tokens if token not in special]
    first_reference = next((i for i, token in enumerate(reference) if token not in special), None)
    reference_waits = reference.count(tokenizer.pad)
    generated_waits = generated.predicted_tokens.count(tokenizer.pad)
    row = dict(metadata or {})
    row.update(
        {
            'prediction_mode': 'autoregressive',
            'reference_ipa': tokenizer.decode(reference_content).strip(),
            'generated_ipa': tokenizer.decode(predicted_content).strip(),
            'exact_phoneme_match': reference_content == predicted_content,
            'target_first_phoneme_step': (
                None if first_reference is None else fixed_delay + max(0, first_reference - 1) + 1
            ),
            'generated_first_phoneme_step': generated.first_phoneme_step,
            'target_wait_percent': 100 * reference_waits / max(1, reference_waits + len(reference_content)),
            'generated_wait_percent': 100 * generated_waits / max(1, generated_waits + len(predicted_content)),
            'generated_steps': generated.decoder_steps,
            'generated_wait_tokens': generated_waits,
            'reached_eos': generated.reached_eos,
            'truncated': not generated.reached_eos,
            'no_phoneme': not predicted_content,
        }
    )
    return row


def teacher_forced_phoneme_tokens(predictions, lengths, tokenizer):
    """Unstack B,S,T argmax IDs and prepend the input BOS for timing statistics.

    The horizon comes from ground-truth lengths. A predicted EOS never shortens
    this horizon: later predictions still use ground-truth history.
    """
    batch, stacking_factor, _ = predictions.shape
    ids = predictions.transpose(1, 2).reshape(batch, -1)
    bos = ids.new_full((batch, 1), tokenizer.bos_token_id)
    return torch.cat([bos, ids], dim=1), lengths * stacking_factor + 1


def teacher_forced_phoneme_example(
    reference_tokens, predicted_tokens, tokenizer, *, fixed_delay=0, stacking_factor=1, metadata=None
):
    """Describe existing predictions without claiming free-running generation."""
    row = tuple(predicted_tokens.detach().cpu().tolist())
    special = {tokenizer.pad, tokenizer.bos_token_id, tokenizer.eos_token_id}
    first = next((fixed_delay + i // stacking_factor + 1 for i, token in enumerate(row) if token not in special), None)
    generated = SimpleNamespace(
        predicted_tokens=row,
        first_phoneme_step=first,
        decoder_steps=len(row) // stacking_factor,
        reached_eos=tokenizer.eos_token_id in row,
    )
    example = phoneme_decoding_example(
        reference_tokens, generated, tokenizer, fixed_delay=fixed_delay, metadata=metadata
    )
    reference = reference_tokens.detach().cpu().tolist()
    first_reference = next((i for i, token in enumerate(reference) if token not in special), None)
    example['target_first_phoneme_step'] = (
        None if first_reference is None else fixed_delay + max(0, first_reference - 1) // stacking_factor + 1
    )
    example.update(prediction_mode='teacher_forced', truncated=None)
    return example


def phoneme_delay_counts(tokens, lengths, tokenizer, fixed_delay=0, stacking_factor=1):
    """Return additive statistics and maximum for unpadded, BOS-led sequences.

    Statistics: sequences, sequences with phonemes, sum of one-based first
    phoneme steps, PADs, PADs + real phonemes, sequences with EOS, and sum of
    per-sequence wait fractions. BOS/EOS and right padding never enter coverage.
    The initial BOS is an input; a phoneme immediately after it is predicted
    on step one. An absent phoneme contributes no artificial zero latency.
    """
    valid_length = torch.arange(tokens.size(1), device=tokens.device)[None, :] < lengths[:, None]
    real = valid_length.clone()
    for special in (tokenizer.pad, tokenizer.bos_token_id, tokenizer.eos_token_id):
        real &= tokens.ne(special)
    pad = valid_length & tokens.eq(tokenizer.pad)
    has_first = real.any(dim=1)
    first_index = real.to(torch.int64).argmax(dim=1)
    steps = fixed_delay + (first_index - 1).clamp_min(0) // stacking_factor + 1
    steps = torch.where(has_first, steps, 0)
    pads = pad.sum(dim=1)
    content = pads + real.sum(dim=1)
    eos = (valid_length & tokens.eq(tokenizer.eos_token_id)).any(dim=1)
    counts = torch.stack(
        [
            lengths.new_tensor(tokens.size(0)),
            has_first.sum(),
            steps.sum(),
            pads.sum(),
            content.sum(),
            eos.sum(),
            (pads.double() / content.clamp_min(1)).sum(),
        ]
    ).double()
    return counts, steps.max().double()


def phoneme_delay_metrics(counts, maximum, prefix):
    """Convert globally summed counts and globally reduced maximum to metrics."""
    sequences, with_phonemes, first_sum, pads, content, eos, fraction_sum = counts
    metrics = {
        f'{prefix}_samples': sequences,
        f'{prefix}_wait_percent': 100 * pads / content.clamp_min(1),
        f'{prefix}_wait_percent_mean': 100 * fraction_sum / sequences.clamp_min(1),
        f'{prefix}_no_phoneme_percent': 100 * (sequences - with_phonemes) / sequences.clamp_min(1),
        f'{prefix}_eos_percent': 100 * eos / sequences.clamp_min(1),
    }
    if with_phonemes > 0:
        metrics[f'{prefix}_first_phoneme_step'] = first_sum / with_phonemes
        metrics[f'{prefix}_first_phoneme_step_max'] = maximum
    return metrics
