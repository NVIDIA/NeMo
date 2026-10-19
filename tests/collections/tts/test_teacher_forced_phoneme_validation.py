# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast teacher-forced timing/PAD diagnostics require no autoregressive calls."""

import datetime
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.parts.utils import phoneme_delay_metrics as metrics

pytestmark = pytest.mark.unit
TOKENIZER = SimpleNamespace(pad=0, bos_token_id=80, eos_token_id=81, decode=lambda ids: ' '.join(map(str, ids)))


def model_for_stats(delay=8, dynamic=True):
    return SimpleNamespace(
        cfg={
            'dynamic_phoneme_delay': dynamic,
            'text_only_val_ar_max_samples': 0,
            'text_only_val_wandb_max_examples': 16,
        },
        phoneme_tokenizer=TOKENIZER,
        phoneme_stacking_factor=1,
        training_modes=[SimpleNamespace(streaming_phonemes_delay=delay)],
        _text_only_val_tf_examples=None,
        _text_only_val_examples=None,
        device='cpu',
        forwards=[],
        trainer=SimpleNamespace(sanity_checking=False),
        loggers=[],
        global_rank=0,
        global_step=42,
        current_epoch=1,
    )


def test_reference_horizon_masks_padding_but_does_not_stop_at_predicted_eos():
    model = model_for_stats()
    # The final token of row one and later tokens of row two are batch padding.
    predictions = torch.tensor([[[0, 0, 7, 81, 0, 99]], [[81, 8, 0, 0, 99, 99]]])
    lengths = torch.tensor([5, 2])
    result = EasyMagpieTTSModel._text_only_teacher_forced_delay_stats(model, {}, predictions, lengths)
    counts = result['val_text_only_teacher_forced_delay_counts']
    stats = metrics.phoneme_delay_metrics(counts, result['val_text_only_teacher_forced_delay_max'], 'tf')
    assert stats['tf_samples'] == 2
    assert stats['tf_first_phoneme_step'] == 2.5  # first real steps 3 and 2, even after predicted EOS
    assert stats['tf_wait_percent'] == 60  # 3 PADs / (3 PADs + 2 phonemes)
    assert stats['tf_wait_percent_mean'] == 37.5
    assert stats['tf_eos_percent'] == 100
    assert not model.forwards


@pytest.mark.parametrize('dynamic,expected', [(True, 3), (False, 11)])
def test_fixed_delay_and_missing_phonemes_have_no_fake_zero_latency(dynamic, expected):
    model = model_for_stats(dynamic=dynamic)
    predictions = torch.tensor([[[0, 0, 7, 81]], [[0, 0, 0, 81]]])
    result = EasyMagpieTTSModel._text_only_teacher_forced_delay_stats(model, {}, predictions, torch.tensor([4, 4]))
    counts = result['val_text_only_teacher_forced_delay_counts']
    stats = metrics.phoneme_delay_metrics(counts, result['val_text_only_teacher_forced_delay_max'], 'tf')
    assert stats['tf_first_phoneme_step'] == expected
    assert stats['tf_no_phoneme_percent'] == 50


def test_stacked_predictions_follow_decoder_steps_not_flattened_channel_order():
    model = model_for_stats()
    model.phoneme_stacking_factor = 2
    predictions = torch.tensor([[[0, 7, 81], [0, 8, 81]]])
    result = EasyMagpieTTSModel._text_only_teacher_forced_delay_stats(model, {}, predictions, torch.tensor([3]))
    assert result['val_text_only_teacher_forced_delay_counts'][2] == 2
    assert result['val_text_only_teacher_forced_delay_counts'][3] == 2


def test_teacher_forced_wandb_rows_exist_with_ar_disabled_and_no_extra_forward():
    model = model_for_stats()
    logger = Mock(spec=WandbLogger)
    model.loggers = [logger]
    EasyMagpieTTSModel._prepare_text_only_validation_examples(model)
    assert model._text_only_val_examples is None
    assert model._text_only_val_tf_examples == []
    predictions = torch.tensor([[[0, 7, 81, 8]]])
    data = {
        'phoneme_tokens': torch.tensor([[80, 0, 7, 8, 81]]),
        'phoneme_tokens_lens': torch.tensor([5]),
        'input_texts': ['$10'],
        'raw_texts': ['$10'],
        'sample_id': ['example'],
    }
    EasyMagpieTTSModel._text_only_teacher_forced_delay_stats(model, data, predictions, torch.tensor([4]))
    row = model._text_only_val_tf_examples[0]
    assert row['prediction_mode'] == 'teacher_forced' and row['truncated'] is None
    assert row['generated_ipa'] == row['reference_ipa'] == '7 8'  # retains teacher-forced token after predicted EOS
    assert row['generated_first_phoneme_step'] == 2 and row['generated_steps'] == 4
    assert not model.forwards
    EasyMagpieTTSModel._log_text_only_validation_examples(model)
    assert logger.log_table.call_count == 1
    assert logger.log_table.call_args.args[0] == 'val/text_only_teacher_forced_examples'
    assert logger.log_table.call_args.kwargs['step'] == 42


@pytest.mark.parametrize('name', ['easy_magpietts_lhotse.yaml', 'easy_magpietts_lhotse_multiturn.yaml'])
def test_default_configs_disable_autoregression(name):
    config = OmegaConf.load(Path(__file__).resolve().parents[3] / 'examples/tts/conf/magpietts' / name)
    assert config.model.text_only_val_ar_max_samples == 0


def tf_rank_worker(rank, rendezvous, directory):
    torch.distributed.init_process_group(
        'gloo', init_method='file://' + rendezvous, rank=rank, world_size=2, timeout=datetime.timedelta(seconds=30)
    )
    try:
        model = model_for_stats()
        logger = Mock(spec=WandbLogger)
        captured_rows = []
        logger.log_table.side_effect = lambda key, **kwargs: captured_rows.append(key)
        model.loggers = [logger] if rank == 0 else []
        model.global_rank = rank
        EasyMagpieTTSModel._prepare_text_only_validation_examples(model)
        # Rank zero sees only speech; rank one supplies all text statistics/rows.
        result = {}
        if rank == 1:
            data = {'phoneme_tokens': torch.tensor([[80, 0, 7, 81]]), 'phoneme_tokens_lens': torch.tensor([4])}
            result = EasyMagpieTTSModel._text_only_teacher_forced_delay_stats(
                model, data, torch.tensor([[[0, 7, 81]]]), torch.tensor([3])
            )
        model.validation_step_outputs = [result]
        captured = {}
        model.log_dict = lambda values, **kwargs: captured.update({key: float(value) for key, value in values.items()})
        EasyMagpieTTSModel._log_text_only_validation_delay_metrics(model)
        EasyMagpieTTSModel._log_text_only_validation_examples(model)
        (Path(directory) / f'{rank}.json').write_text(json.dumps(dict(metrics=captured, uploads=captured_rows)))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(sys.platform == 'win32', reason='Uses fork in the CPU container')
def test_teacher_forcing_reduces_and_logs_when_rank_zero_only_has_speech(tmp_path):
    torch.multiprocessing.start_processes(
        tf_rank_worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=2, start_method='fork', join=True
    )
    for rank in range(2):
        result = json.loads((tmp_path / f'{rank}.json').read_text())
        assert result['metrics']['val/text_only_teacher_forced_first_phoneme_step'] == 2
        assert result['metrics']['val/text_only_teacher_forced_wait_percent'] == 50
        assert result['uploads'] == (['val/text_only_teacher_forced_examples'] if rank == 0 else [])
