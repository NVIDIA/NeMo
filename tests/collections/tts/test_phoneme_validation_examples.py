# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W&B decoding rows reuse generated IDs and preserve actual loader selections."""

import datetime
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger

from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.parts.utils.phoneme_delay_metrics import TEXT_ONLY_EXAMPLE_COLUMNS, phoneme_decoding_example
from tests.collections.tts.data.test_dynamic_phoneme_delay import cuts, dataset, row
from tests.collections.tts.test_phoneme_delay_metrics import batch, make_decoder

pytestmark = pytest.mark.unit


def tokenizer():
    return SimpleNamespace(
        pad=0,
        bos_token_id=80,
        eos_token_id=81,
        decode=lambda ids: ' '.join({7: 'wʌn', 8: 'tu', 18: 'θɹi', 19: 'fɔɹ'}[value] for value in ids),
    )


@pytest.mark.parametrize('fixed_delay', [0, 8])
def test_example_strips_waits_and_specials_but_keeps_real_delays(fixed_delay):
    generated = SimpleNamespace(
        predicted_tokens=(0, 7, 8, 81), decoder_steps=4, first_phoneme_step=fixed_delay + 2, reached_eos=True
    )
    example = phoneme_decoding_example(
        torch.tensor([80, 0, 0, 7, 8, 81]), generated, tokenizer(), fixed_delay=fixed_delay
    )
    assert example['reference_ipa'] == example['generated_ipa'] == 'wʌn tu'
    assert example['exact_phoneme_match']
    assert example['target_first_phoneme_step'] == fixed_delay + 3
    assert example['generated_first_phoneme_step'] == fixed_delay + 2
    assert example['target_wait_percent'] == 50
    assert example['generated_wait_percent'] == pytest.approx(100 / 3)
    assert example['generated_wait_tokens'] == 1
    assert not example['truncated'] and not example['no_phoneme']


@pytest.mark.parametrize('predicted,eos', [((0, 0, 0), False), ((81,), True)])
def test_empty_and_truncated_outputs_are_logged_without_fake_latency(predicted, eos):
    generated = SimpleNamespace(
        predicted_tokens=predicted, decoder_steps=len(predicted), first_phoneme_step=None, reached_eos=eos
    )
    example = phoneme_decoding_example(torch.tensor([80, 7, 81]), generated, tokenizer())
    assert example['generated_ipa'] == ''
    assert example['no_phoneme'] and example['generated_first_phoneme_step'] is None
    assert example['truncated'] == (not eos)
    assert not example['exact_phoneme_match']


def test_row_limit_does_not_limit_metrics_or_generate_more_tokens():
    model = make_decoder([0, 7, 81] * 2)
    model.cfg.text_only_val_ar_max_samples = None
    model.cfg.text_only_val_ar_max_steps = None
    model.phoneme_tokenizer = tokenizer()
    model._text_only_val_ar_samples = 0
    model._text_only_val_examples = []
    model._text_only_val_examples_limit = 1
    model.trainer = SimpleNamespace(sanity_checking=False)
    model.global_step, model.current_epoch, model.global_rank = 50, 2, 0
    data = batch()
    data.update(
        input_texts=['ten dollars'],
        raw_texts=['$10'],
        sample_id=['sample-1'],
        dataset_names=['numbers'],
        languages=['en-US'],
        text_tokenizer_names=['text'],
    )
    for _ in range(2):
        stats = EasyMagpieTTSModel._text_only_validation_delay_stats(model, data)
        assert stats['val_text_only_ar_delay_counts'][0] == 1
    assert len(model.forwards) == 6 and model._text_only_val_ar_samples == 2
    assert len(model._text_only_val_examples) == 1
    example = model._text_only_val_examples[0]
    assert example['input_text'] == 'ten dollars' and example['raw_text'] == '$10'
    assert example['reference_ipa'] == 'θɹi fɔɹ' and example['generated_ipa'] == 'wʌn'
    assert example['sample_id'] == 'sample-1' and example['global_step'] == 50
    assert example['target_first_phoneme_step'] == 3 and example['generated_first_phoneme_step'] == 2


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('normalized', [0.0, 1.0])
def test_loader_logs_actual_chosen_input_without_resampling(tmp_path, enabled, normalized):
    data = dataset(enabled, normalized)[cuts(tmp_path, [row()])]
    expected = 'one next' if normalized else '$12345 next'
    assert data['input_texts'] == [expected]
    assert data['raw_texts'] == ['$12345 next']
    assert data['text_tokenizer_names'] == ['text']
    ds = dataset(enabled, normalized)
    ids = data['text'][0, : int(data['text_lens'][0])].tolist()
    assert ''.join(chr(value - 1) for value in ids if value not in {ds.pad_id, ds.bos_id, ds.eos_id}) == expected


@pytest.mark.parametrize('kind', ['no_logger', 'disabled', 'no_ar', 'sanity'])
def test_no_wandb_or_disabled_collection_has_no_rows(kind):
    model = SimpleNamespace(
        cfg={'text_only_val_ar_max_samples': None, 'text_only_val_wandb_max_examples': 16},
        device='cpu',
        loggers=[Mock(spec=WandbLogger)] if kind != 'no_logger' else [],
        trainer=SimpleNamespace(sanity_checking=kind == 'sanity'),
    )
    if kind == 'disabled':
        model.cfg['text_only_val_wandb_max_examples'] = 0
    if kind == 'no_ar':
        model.cfg['text_only_val_ar_max_samples'] = 0
    EasyMagpieTTSModel._prepare_text_only_validation_examples(model)
    assert model._text_only_val_examples is None
    EasyMagpieTTSModel._log_text_only_validation_examples(model)


def test_actual_wandb_table_api_gets_current_step_without_creating_a_run(tmp_path, monkeypatch):
    monkeypatch.setattr(wandb, 'init', lambda *a, **k: pytest.fail('Unit test must never create a W&B run'))
    captured = {}

    def capture(logger, metrics, step=None):
        captured.update(metrics=metrics, step=step)

    monkeypatch.setattr(WandbLogger, 'log_metrics', capture)
    logger = WandbLogger(project='unit-tests', save_dir=str(tmp_path), offline=True)
    model = SimpleNamespace(
        cfg={'text_only_val_wandb_max_examples': 1},
        global_rank=0,
        global_step=42,
        loggers=[logger],
        _text_only_val_examples=[{'sample_id': 'first'}, {'sample_id': 'second'}],
    )
    EasyMagpieTTSModel._log_text_only_validation_examples(model)
    table = captured['metrics']['val/text_only_decoding_examples']
    assert isinstance(table, wandb.Table) and table.columns == TEXT_ONLY_EXAMPLE_COLUMNS
    assert len(table.data) == 1 and table.data[0][table.columns.index('sample_id')] == 'first'
    assert captured['step'] == 42 and model._text_only_val_examples is None


def test_collection_resets_each_validation_interval():
    model = SimpleNamespace(
        cfg={'text_only_val_ar_max_samples': None, 'text_only_val_wandb_max_examples': 2},
        device='cpu',
        loggers=[Mock(spec=WandbLogger)],
        trainer=SimpleNamespace(sanity_checking=False),
    )
    EasyMagpieTTSModel._prepare_text_only_validation_examples(model)
    model._text_only_val_examples.append({'sample_id': 'old'})
    EasyMagpieTTSModel._prepare_text_only_validation_examples(model)
    assert model._text_only_val_examples == [] and model._text_only_val_examples_limit == 2


def example_rank_worker(rank, rendezvous, directory, empty_root):
    torch.distributed.init_process_group(
        'gloo', init_method='file://' + rendezvous, rank=rank, world_size=2, timeout=datetime.timedelta(seconds=30)
    )
    try:
        captured = []
        logger = Mock(spec=WandbLogger)
        logger.log_table.side_effect = lambda key, **kwargs: captured.append(kwargs)
        model = SimpleNamespace(
            cfg={'text_only_val_ar_max_samples': None, 'text_only_val_wandb_max_examples': 2},
            device='cpu',
            loggers=[logger] if rank == 0 else [],
            trainer=SimpleNamespace(sanity_checking=False),
            global_rank=rank,
            global_step=77,
        )
        EasyMagpieTTSModel._prepare_text_only_validation_examples(model)
        assert model._text_only_val_examples is not None
        assert model._text_only_val_examples_limit == 1
        if rank == 1 or not empty_root:
            model._text_only_val_examples.append({'sample_id': f'rank-{rank}', 'rank': rank})
        EasyMagpieTTSModel._log_text_only_validation_examples(model)
        result = {'uploads': len(captured)}
        if captured:
            table = captured[0]
            result['ids'] = [r[TEXT_ONLY_EXAMPLE_COLUMNS.index('sample_id')] for r in table['data']]
            result['step'] = table['step']
        (Path(directory) / f'{rank}.json').write_text(json.dumps(result))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(sys.platform == 'win32', reason='Uses fork in the CPU container')
@pytest.mark.parametrize('empty_root', [False, True])
def test_distributed_examples_upload_once_including_text_from_nonzero_ranks(tmp_path, empty_root):
    torch.multiprocessing.start_processes(
        example_rank_worker,
        args=(str(tmp_path / 'rendezvous'), str(tmp_path), empty_root),
        nprocs=2,
        start_method='fork',
        join=True,
    )
    primary = json.loads((tmp_path / '0.json').read_text())
    assert primary['uploads'] == 1 and primary['step'] == 77
    assert primary['ids'] == (['rank-1'] if empty_root else ['rank-0', 'rank-1'])
    assert json.loads((tmp_path / '1.json').read_text())['uploads'] == 0
