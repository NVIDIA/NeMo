# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual cached decoding, schedule timing, and distributed validation metrics."""

import datetime
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger

from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.modules.magpietts_inference.phoneme_frontend import generate_text_only_phoneme_tokens
from nemo.collections.tts.modules.magpietts_modules import LocalTransformerType
from nemo.collections.tts.parts.utils.phoneme_delay_metrics import phoneme_delay_counts, phoneme_delay_metrics
from tests.collections.tts.modules.magpietts_inference.test_phoneme_frontend import TinyFrontend

pytestmark = pytest.mark.unit
TOKENIZER = SimpleNamespace(pad=0, bos_token_id=80, eos_token_id=81)


def test_schedule_excludes_context_specials_and_right_padding():
    ids = torch.tensor([[80, 0, 0, 7, 0, 8, 81, 0], [80, 9, 81, 0, 0, 0, 0, 0]])
    counts, maximum = phoneme_delay_counts(ids, torch.tensor([7, 3]), TOKENIZER)
    metrics = phoneme_delay_metrics(counts, maximum, 'test')
    assert metrics['test_first_phoneme_step'] == 2  # steps 3 and 1
    assert metrics['test_first_phoneme_step_max'] == 3
    assert metrics['test_wait_percent'] == 50  # 3 waits / (3 waits + 3 phonemes)
    assert metrics['test_wait_percent_mean'] == 30  # (60% + 0%) / 2
    assert metrics['test_no_phoneme_percent'] == 0
    assert metrics['test_eos_percent'] == 100


@pytest.mark.parametrize('delay', [0, 3, 8])
def test_fixed_delay_is_counted_before_first_phoneme(delay):
    counts, maximum = phoneme_delay_counts(torch.tensor([[80, 7, 81]]), torch.tensor([3]), TOKENIZER, delay)
    assert counts[2] == delay + 1
    assert maximum == delay + 1
    assert counts[3] == 0


def test_stacking_counts_decoder_steps_not_flat_token_indices():
    counts, maximum = phoneme_delay_counts(
        torch.tensor([[80, 0, 0, 7, 81]]), torch.tensor([5]), TOKENIZER, stacking_factor=2
    )
    assert counts[2] == maximum == 2


def test_no_phoneme_never_contributes_zero_latency():
    counts, maximum = phoneme_delay_counts(
        torch.tensor([[80, 0, 0, 81], [80, 7, 81, 0]]), torch.tensor([4, 3]), TOKENIZER
    )
    metrics = phoneme_delay_metrics(counts, maximum, 'test')
    assert metrics['test_first_phoneme_step'] == 1
    assert metrics['test_no_phoneme_percent'] == 50
    counts, maximum = phoneme_delay_counts(torch.tensor([[80, 0, 0]]), torch.tensor([3]), TOKENIZER)
    metrics = phoneme_delay_metrics(counts, maximum, 'test')
    assert 'test_first_phoneme_step' not in metrics
    assert metrics['test_wait_percent'] == 100
    assert metrics['test_no_phoneme_percent'] == 100
    assert metrics['test_eos_percent'] == 0


def make_decoder(predictions, *, dynamic=True, delay=8, repeat=False):
    model = TinyFrontend(delay=delay)
    model.phoneme_tokenizer.pad = 0
    model.cfg.dynamic_phoneme_delay = dynamic
    model.repeat_context = repeat
    model._repeat_gated_context_embeddings = Mock(side_effect=lambda c, l, p: torch.ones(1, p.numel(), 4))
    ids = iter(predictions)

    def project(hidden):
        logits = torch.full((1, 1, 82), -100.0)
        logits[..., next(ids)] = 100.0
        return logits

    model.phoneme_final_proj = project
    return model.eval()


@pytest.mark.parametrize('dynamic', [True, False])
@pytest.mark.parametrize('repeat', [True, False])
def test_greedy_delay_uses_predicted_waits_and_excludes_prefix(dynamic, repeat):
    model = make_decoder([0, 0, 7, 0, 8, 81], dynamic=dynamic, repeat=repeat)
    result = generate_text_only_phoneme_tokens(
        model, torch.tensor([[11, 12, 13]]), torch.tensor([[21] * 9]), model.training_modes[0], 6
    )
    assert result.first_phoneme_step == (3 if dynamic else 11)
    assert result.decoder_steps == 6 and result.reached_eos
    assert result.predicted_tokens == (0, 0, 7, 0, 8, 81)
    assert result.wait_tokens == (3 if dynamic else 0)
    assert result.token_ids == ((7, 8) if dynamic else (0, 0, 7, 0, 8))
    assert all(call['use_cache'] for call in model.forwards)
    assert model.tokenizer.encode.call_count == 0
    assert model._repeat_gated_context_embeddings.called == repeat


@pytest.mark.parametrize('predictions,eos', [([0, 0, 0], False), ([0, 81], True)])
def test_greedy_missing_phoneme_and_truncation_are_explicit(predictions, eos):
    model = make_decoder(predictions)
    result = generate_text_only_phoneme_tokens(
        model, torch.tensor([[11, 13]]), torch.tensor([[21]]), model.training_modes[0], len(predictions)
    )
    assert result.first_phoneme_step is None
    assert result.reached_eos == eos
    assert result.decoder_steps == len(predictions)
    assert len(model.forwards) == len(predictions)


def batch():
    return {
        'text': torch.tensor([[11, 12, 13, 0]]),
        'text_lens': torch.tensor([3]),
        'context_text_tokens': torch.tensor([[21, 0]]),
        'context_text_tokens_lens': torch.tensor([1]),
        'phoneme_tokens': torch.tensor([[80, 0, 0, 18, 19, 81, 0]]),
        'phoneme_tokens_lens': torch.tensor([6]),
    }


def test_validation_reuses_selected_tokens_and_never_teacher_forces_ar():
    model = make_decoder([0, 7, 81])
    model.cfg.text_only_val_ar_max_samples = 1
    model.cfg.text_only_val_ar_max_steps = 3
    model._text_only_val_ar_samples = 0
    model.trainer = SimpleNamespace(sanity_checking=False)
    result = EasyMagpieTTSModel._text_only_validation_delay_stats(model, batch())
    assert result['val_text_only_target_delay_counts'][2] == 3
    assert result['val_text_only_ar_delay_counts'][2] == 2
    assert result['val_text_only_ar_delay_counts'][3] == 1
    assert result['val_text_only_ar_delay_counts'][5] == 1
    assert [int(t.flatten()[0]) for t in model.phoneme_inputs] == [80, 0, 7]
    assert model._text_only_val_ar_samples == 1
    # Budget applies across the entire interval, not independently per batch.
    again = EasyMagpieTTSModel._text_only_validation_delay_stats(model, batch())
    assert again['val_text_only_ar_delay_counts'][0] == 0
    assert len(model.forwards) == 3


@pytest.mark.parametrize('sanity,budget', [(False, 0), (True, 2)])
def test_disabled_and_sanity_checks_do_not_run_generation(sanity, budget):
    model = make_decoder([])
    model.cfg.text_only_val_ar_max_samples = budget
    model.trainer = SimpleNamespace(sanity_checking=sanity)
    result = EasyMagpieTTSModel._text_only_validation_delay_stats(model, batch())
    assert 'val_text_only_target_delay_counts' in result
    assert 'val_text_only_ar_delay_counts' not in result
    assert not model.forwards


def test_null_budget_evaluates_every_sample_across_every_validation_batch():
    model = make_decoder([0, 7, 81] * 6)
    model.cfg.text_only_val_ar_max_samples = None
    model.cfg.text_only_val_ar_max_steps = None
    model._text_only_val_ar_samples = 0
    model.trainer = SimpleNamespace(sanity_checking=False)
    examples = {key: value.repeat((3,) + (1,) * (value.ndim - 1)) for key, value in batch().items()}
    for _ in range(2):
        result = EasyMagpieTTSModel._text_only_validation_delay_stats(model, examples)
        assert result['val_text_only_ar_delay_counts'][0] == 3
        assert result['val_text_only_ar_delay_counts'][2] == 6
    assert model._text_only_val_ar_samples == 6
    assert len(model.forwards) == 18


def test_automatic_step_limit_still_stops_endless_padding():
    model = make_decoder([0] * 256)
    result = generate_text_only_phoneme_tokens(
        model, torch.tensor([[11, 13]]), torch.tensor([[21]]), model.training_modes[0], None
    )
    assert result.decoder_steps == 256
    assert not result.reached_eos and result.first_phoneme_step is None
    assert result.wait_tokens == 256


def test_automatic_step_limit_allows_complete_outputs_longer_than_2048():
    model = make_decoder([7] * 2050 + [81])
    result = generate_text_only_phoneme_tokens(
        model, torch.tensor([[11] * 299 + [13]]), torch.tensor([[21]]), model.training_modes[0], None
    )
    assert result.decoder_steps == 2051
    assert result.reached_eos and len(result.token_ids) == 2050


@pytest.mark.parametrize('cap', [0, -1, True, 1.5])
def test_invalid_explicit_generation_caps_fail_clearly(cap):
    model = make_decoder([])
    with pytest.raises(ValueError, match='null or positive max_steps'):
        generate_text_only_phoneme_tokens(
            model, torch.tensor([[11, 13]]), torch.tensor([[21]]), model.training_modes[0], cap
        )


@pytest.mark.parametrize('name', ['easy_magpietts_lhotse.yaml', 'easy_magpietts_lhotse_multiturn.yaml'])
def test_lhotse_configs_default_to_full_evaluation_with_adaptive_caps(name):
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(__file__).resolve().parents[3] / 'examples/tts/conf/magpietts' / name)
    assert config.model.text_only_val_ar_max_samples is None
    assert config.model.text_only_val_ar_max_steps is None


def test_packed_turns_are_not_silently_scored_as_a_single_turn():
    model = make_decoder([])
    model.cfg.text_only_val_ar_max_samples = 1
    model._text_only_val_ar_samples = 0
    model.trainer = SimpleNamespace(sanity_checking=False)
    packed = batch()
    packed['text'][0, 1] = model.eos_id
    result = EasyMagpieTTSModel._text_only_validation_delay_stats(model, packed)
    assert result['val_text_only_target_delay_counts'][0] == 1
    assert result['val_text_only_ar_skipped'] == 1
    assert not model.forwards


def distributed_worker(rank, rendezvous, directory):
    torch.distributed.init_process_group(
        'gloo', init_method='file://' + rendezvous, rank=rank, world_size=2, timeout=datetime.timedelta(seconds=30)
    )
    try:
        counts, maximum = phoneme_delay_counts(torch.tensor([[80, 0, 0, 7, 81]]), torch.tensor([5]), TOKENIZER)
        # Rank one receives only speech. It must still participate in reductions.
        outputs = (
            [{'val_text_only_target_delay_counts': counts, 'val_text_only_target_delay_max': maximum}]
            if rank == 0
            else [{}]
        )
        captured = {}
        model = SimpleNamespace(
            device='cpu',
            cfg={},
            validation_step_outputs=outputs,
            log_dict=lambda metrics, **kw: captured.update({k: float(v) for k, v in metrics.items()}),
        )
        EasyMagpieTTSModel._log_text_only_validation_delay_metrics(model)
        (Path(directory) / f'{rank}.json').write_text(json.dumps(captured))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(sys.platform == 'win32', reason='Uses fork in the CPU container')
def test_mixed_distributed_ranks_reduce_counts_and_maxima_without_deadlock(tmp_path):
    torch.multiprocessing.start_processes(
        distributed_worker,
        args=(str(tmp_path / 'rendezvous'), str(tmp_path)),
        nprocs=2,
        start_method='fork',
        join=True,
    )
    for rank in range(2):
        metrics = json.loads((tmp_path / f'{rank}.json').read_text())
        assert metrics['val/text_only_target_samples'] == 1
        assert metrics['val/text_only_target_first_phoneme_step'] == 3
        assert metrics['val/text_only_target_first_phoneme_step_max'] == 3
        assert metrics['val/text_only_target_wait_percent'] == pytest.approx(200 / 3)


def test_epoch_ratios_are_weighted_by_tokens_not_batch_percentages(monkeypatch):
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
    outputs = []
    for ids in [[80, 0, 0, 7, 81], [80, 7, 7, 7, 7, 7, 7, 81]]:
        counts, maximum = phoneme_delay_counts(torch.tensor([ids]), torch.tensor([len(ids)]), TOKENIZER)
        outputs.append({'val_text_only_target_delay_counts': counts, 'val_text_only_target_delay_max': maximum})
    captured = {}
    model = SimpleNamespace(
        device='cpu', cfg={}, validation_step_outputs=outputs, log_dict=lambda metrics, **kw: captured.update(metrics)
    )
    EasyMagpieTTSModel._log_text_only_validation_delay_metrics(model)
    assert captured['val/text_only_target_wait_percent'] == pytest.approx(200 / 9)
    assert captured['val/text_only_target_first_phoneme_step'] == 2


def test_real_validation_step_and_epoch_metrics_reach_lightning_logger(tmp_path):
    import csv

    class ValidationHarness(LightningModule):
        validation_step = EasyMagpieTTSModel.validation_step
        on_validation_epoch_start = EasyMagpieTTSModel.on_validation_epoch_start
        on_validation_epoch_end = EasyMagpieTTSModel.on_validation_epoch_end
        _text_only_validation_delay_stats = EasyMagpieTTSModel._text_only_validation_delay_stats
        _log_text_only_validation_delay_metrics = EasyMagpieTTSModel._log_text_only_validation_delay_metrics
        _prepare_text_only_validation_examples = EasyMagpieTTSModel._prepare_text_only_validation_examples
        _log_text_only_validation_examples = EasyMagpieTTSModel._log_text_only_validation_examples
        _is_text_only_batch = staticmethod(EasyMagpieTTSModel._is_text_only_batch)

        def __init__(self):
            super().__init__()
            self.stub = make_decoder([0, 7, 81])
            self.cfg = self.stub.cfg
            self.cfg.text_only_val_ar_max_samples = None
            self.cfg.text_only_val_ar_max_steps = None
            for key in (
                'training_modes',
                'phoneme_tokenizer',
                'phoneme_stacking_factor',
                'task_embedding',
                'repeat_context',
                'disable_cas_for_context_text',
                'pad_id',
                'eos_id',
            ):
                setattr(self, key, getattr(self.stub, key))
            self.phoneme_vocab_size = 82
            self.run_val_inference = False
            self.local_transformer_type = LocalTransformerType.NO_LT
            self.validation_step_outputs = []

        def _process_text_only_batch(self, *args, **kwargs):
            targets = args[4][:, : int(args[5].max())][:, None, 1:]
            logits = torch.nn.functional.one_hot(targets[:, 0], 82).float()
            return SimpleNamespace(
                loss=torch.tensor(0.1),
                phoneme_loss=torch.tensor(0.1),
                codebook_loss=torch.tensor(0.0),
                phoneme_tokens_target=targets,
                phoneme_tokens_lens_target=args[5] - 1,
                phoneme_logits=logits,
            )

        def embed_text_tokens(self, *args, **kwargs):
            return self.stub.embed_text_tokens(*args, **kwargs)

        def embed_phoneme_tokens(self, *args, **kwargs):
            return self.stub.embed_phoneme_tokens(*args, **kwargs)

        def phoneme_final_proj(self, *args, **kwargs):
            return self.stub.phoneme_final_proj(*args, **kwargs)

        def forward(self, **kwargs):
            return self.stub.forward(**kwargs)

    model = ValidationHarness()
    logger = CSVLogger(str(tmp_path), name='delay')
    trainer = Trainer(
        accelerator='cpu',
        devices=1,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    data = batch()
    data['task'] = 'text_only'
    trainer.validate(model, torch.utils.data.DataLoader([data], batch_size=None))
    with open(Path(logger.log_dir) / 'metrics.csv', newline='') as handle:
        row = list(csv.DictReader(handle))[-1]
    assert float(row['val/text_only_target_first_phoneme_step']) == 3
    assert float(row['val/text_only_ar_first_phoneme_step']) == 2
    assert float(row['val/text_only_target_wait_percent']) == 50
    assert float(row['val/text_only_ar_wait_percent']) == 50
    assert float(row['val/text_only_ar_eos_percent']) == 100
    assert float(row['val/text_only_phoneme_token_accuracy']) == 1
    assert not model.validation_step_outputs
