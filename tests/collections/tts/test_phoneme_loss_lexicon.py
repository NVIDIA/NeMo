# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for matching, target alignment, and weighted loss gradients."""

import csv
import datetime
import json
import pickle
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import yaml
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch import nn

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import IPABPETokenizer
from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import (
    MagpieTTSLhotseMultiturnDataset,
    build_phoneme_channel,
    collate_phoneme_channel,
)
from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
from nemo.collections.tts.parts.utils.phoneme_loss_lexicon import (
    PhonemeLossLexicon,
    make_phoneme_loss_weights,
    stack_phoneme_loss_mask,
)


@pytest.fixture
def lexicon_path(tmp_path):
    path = tmp_path / 'lexicon.yaml'
    data = dict(
        schema_version=1,
        languages=['en'],
        entries=[
            dict(word='million', pronunciations={'en-GB': ['mˈɪliən'], 'en-US': ['mˈɪliən']}),
            dict(word='billion', pronunciations={'en-US': ['bˈɪliən']}),
            dict(word='twenty', pronunciations={'en-US': ['twˈɛnti']}),
            dict(word='decimal point', pronunciations={'en-US': ['dˈɛsɪməl pˈɔɪnt']}),
        ],
    )
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding='utf-8')
    return path


@pytest.fixture
def tokenizer(tmp_path):
    backend = Tokenizer(models.BPE(unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        ['ə mˈɪliən dˈɑːlɚz', 'bˈɪliən', 'twˈɛnti', 'dˈɛsɪməl pˈɔɪnt', 'mɪliənz', 'həlˈoʊ'] * 20,
        trainers.BpeTrainer(
            vocab_size=350, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=['<pad>', '<unk>']
        ),
    )
    path = tmp_path / 'tokenizer.json'
    backend.save(str(path))
    return IPABPETokenizer(str(path))


def _cut(ipa='ə mˈɪliən dˈɑːlɚz', text_only=True, language='en', duration=1.0, start=0.0):
    supervision = SimpleNamespace(
        speaker='agent',
        ipa=ipa,
        text='This number is one million dollars',
        start=start,
        custom={},
        has_custom=lambda key: False,
    )
    return SimpleNamespace(
        id='test',
        task='text_only' if text_only else 'tts',
        lang=language,
        supervisions=[supervision],
        duration=duration,
        sampling_rate=16000,
        has_custom=lambda key: key == 'lang',
    )


def _channel(cut, tokenizer, lexicon=None, **kwargs):
    return build_phoneme_channel(
        cut,
        tokenizer,
        0.01,
        {'agent'},
        [],
        pad_id=tokenizer.pad,
        bos_id=tokenizer.bos,
        eos_id=tokenizer.eos,
        phoneme_loss_lexicon=lexicon,
        **kwargs,
    )


@pytest.mark.unit
def test_matches_whole_words_stress_and_phrases(lexicon_path):
    matcher = PhonemeLossLexicon(lexicon_path)
    ipa = 'mɪliən mˌɪliən mˈɪliənz twˈɛnti dˈɛsɪməl pˈɔɪnt dˈɛsɪməl, pˈɔɪnt'
    spans = matcher.find_spans(ipa)
    assert [ipa[start:end] for start, end in spans] == ['mɪliən', 'mˌɪliən', 'twˈɛnti', 'dˈɛsɪməl pˈɔɪnt']
    assert matcher.supports_language('en-US')
    assert matcher.supports_language('en_GB')
    assert not matcher.supports_language('de')


@pytest.mark.unit
def test_mask_uses_full_sentence_bpe_offsets(lexicon_path, tokenizer):
    matcher = PhonemeLossLexicon(lexicon_path)
    ipa = 'ə mˈɪliən dˈɑːlɚz'
    ids, mask = matcher.encode_with_mask(ipa, tokenizer)
    assert ids == tokenizer.encode(ipa)
    _, offsets = tokenizer.encode_with_offsets(ipa)
    span = matcher.find_spans(ipa)[0]
    assert mask == [end > start and start < span[1] and end > span[0] for start, end in offsets]
    assert any(mask) and not all(mask)
    compiled = matcher.tokenized_pronunciations
    matcher.encode_with_mask(ipa, tokenizer)
    assert matcher.tokenized_pronunciations is compiled  # no per-sample lexicon tokenization


@pytest.mark.unit
@pytest.mark.parametrize('text_only', [True, False])
def test_channel_ids_unchanged_and_special_tokens_unweighted(lexicon_path, tokenizer, text_only):
    cut = _cut(text_only=text_only)
    baseline, baseline_dropout = _channel(cut, tokenizer)
    tokens, dropout, mask = _channel(cut, tokenizer, PhonemeLossLexicon(lexicon_path))
    assert torch.equal(tokens, baseline)
    assert dropout == baseline_dropout
    assert mask.dtype == torch.bool and mask.shape == tokens.shape
    assert mask.any()
    assert not mask[(tokens == tokenizer.bos) | (tokens == tokenizer.eos) | (tokens == tokenizer.pad)].any()


@pytest.mark.unit
@pytest.mark.parametrize('text_only', [True, False])
def test_unsupported_language_gets_zero_mask(lexicon_path, tokenizer, text_only):
    cut = _cut(text_only=text_only, language='de')
    tokens, _, mask = _channel(cut, tokenizer, PhonemeLossLexicon(lexicon_path))
    assert torch.equal(tokens, _channel(cut, tokenizer)[0])
    assert not mask.any()


@pytest.mark.unit
def test_audio_turn_truncation_and_dropout_preserve_alignment(lexicon_path, tokenizer):
    matcher = PhonemeLossLexicon(lexicon_path)
    full_tokens, _, full_mask = _channel(_cut(text_only=False), tokenizer, matcher)
    short_tokens, _, short_mask = _channel(_cut(text_only=False, duration=0.04), tokenizer, matcher)
    assert torch.equal(short_tokens, full_tokens[:4])
    assert torch.equal(short_mask, full_mask[:4])
    tokens, dropout, mask = _channel(
        _cut(text_only=False),
        tokenizer,
        matcher,
        apply_turn_dropout=True,
        phoneme_turn_dropout_batch_prob=1.0,
        phoneme_turn_dropout_turn_prob=1.0,
    )
    assert dropout and not mask.any() and (tokens == tokenizer.pad).all()


@pytest.mark.unit
def test_batch_padding_and_multiple_text_turns(lexicon_path, tokenizer):
    cuts = [_cut(), _cut('bˈɪliən')]
    cuts[0].supervisions.append(_cut('twˈɛnti').supervisions[0])
    args = dict(
        frame_length=0.01,
        roles={'agent'},
        ignore_phoneme_languages=[],
        pad_id=tokenizer.pad,
        bos_id=tokenizer.bos,
        eos_id=tokenizer.eos,
    )
    baseline = collate_phoneme_channel(cuts, tokenizer, **args)
    result = collate_phoneme_channel(cuts, tokenizer, **args, phoneme_loss_lexicon=PhonemeLossLexicon(lexicon_path))
    for actual, expected in zip(result[:3], baseline):
        assert torch.equal(actual, expected)
    assert result[3].shape == result[0].shape
    assert not result[3][1, result[1][1] :].any()
    assert not result[3][result[0] == tokenizer.bos].any()
    assert not result[3][result[0] == tokenizer.eos].any()


@pytest.mark.unit
@pytest.mark.parametrize('factor', [1, 2, 3])
@pytest.mark.parametrize('initial_bos', [True, False])
def test_stacking_and_target_shift_follow_production_stack_codes(factor, initial_bos):
    bos, eos = 99, 100
    tokens = torch.tensor(
        [[bos if initial_bos else 0, 2, 3, 4, eos, 0, 0], [bos if initial_bos else 0, 2, 3, eos, 0, 0, 0]]
    )
    mask = (tokens == 2) | (tokens == 4)
    packed, _ = EasyMagpieTTSInferenceModel.stack_codes(
        None, tokens[:, None], torch.tensor([5, 4]), bos, eos, factor, 1
    )
    expected = ((packed == 2) | (packed == 4))[:, :, 1:]
    assert torch.equal(stack_phoneme_loss_mask(mask, tokens, factor, bos), expected)


class _LossHarness:
    compute_phoneme_loss = EasyMagpieTTSModel.compute_phoneme_loss

    def __init__(self, factor=1):
        self.phoneme_stacking_factor = factor
        self.phoneme_vocab_size = 5
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')


@pytest.mark.unit
def test_selected_loss_and_gradients_scale_exactly_and_padding_is_ignored():
    torch.manual_seed(42)
    model = _LossHarness()
    targets = torch.tensor([[[1, 2, 3, 4]], [[2, 1, 0, 0]]])
    lengths = torch.tensor([4, 2])
    baseline_logits = torch.randn(2, 4, 5, requires_grad=True)
    weighted_logits = baseline_logits.detach().clone().requires_grad_()
    weights = torch.tensor([[[1.0, 3.0, 1.0, 1.0]], [[3.0, 1.0, 9.0, 9.0]]])
    baseline, valid = model.compute_phoneme_loss(baseline_logits, targets, lengths)
    weighted, _ = model.compute_phoneme_loss(weighted_logits, targets, lengths, token_weights=weights)
    raw = model.cross_entropy_loss(weighted_logits.permute(0, 2, 1), targets[:, 0])
    torch.testing.assert_close(weighted, (raw * valid[:, 0] * weights[:, 0]).sum() / valid.sum())
    baseline.backward()
    weighted.backward()
    torch.testing.assert_close(weighted_logits.grad, baseline_logits.grad * weights[:, 0, :, None])
    assert not weighted_logits.grad[1, 2:].any()


@pytest.mark.unit
def test_custom_mask_and_stacked_losses():
    model = _LossHarness(2)
    logits = torch.randn(1, 3, 10, requires_grad=True)
    targets = torch.tensor([[[1, 2, 3], [2, 3, 4]]])
    weights = torch.full_like(targets, 4.0, dtype=torch.float)
    custom = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    baseline, _ = model.compute_phoneme_loss(logits, targets, torch.tensor([3]), custom_mask=custom)
    weighted, _ = model.compute_phoneme_loss(
        logits, targets, torch.tensor([3]), custom_mask=custom, token_weights=weights
    )
    torch.testing.assert_close(weighted, 4 * baseline)
    weighted.backward()
    assert not logits.grad[:, 1].any()


@pytest.mark.unit
def test_coverage_counts_use_effective_loss_mask():
    model = _LossHarness()
    logits = torch.randn(2, 4, 5)
    targets = torch.tensor([[[1, 2, 3, 4]], [[2, 1, 0, 0]]])
    weights = torch.tensor([[[1.0, 3.0, 1.0, 1.0]], [[3.0, 1.0, 9.0, 9.0]]])
    custom = torch.tensor([[1, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)
    _, _, coverage = model.compute_phoneme_loss(
        logits, targets, torch.tensor([4, 2]), custom_mask=custom, token_weights=weights, return_coverage=True
    )
    assert [int(value) for value in coverage] == [4, 1, 5]


@pytest.mark.unit
@pytest.mark.parametrize('skipped', [True, False])
def test_logging_emits_counts_and_safe_percent(monkeypatch, skipped):
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
    logged = {}

    def capture(metrics, **kwargs):
        logged.update(metrics)
        assert kwargs['sync_dist'] is False and kwargs['rank_zero_only'] is True

    model = SimpleNamespace(cfg={'phoneme_lexicon_loss_scale': 2.0}, log_dict=capture)
    output = SimpleNamespace(
        loss=torch.tensor(0.0),
        phoneme_loss_coverage=tuple(torch.tensor(x) for x in ((7, 0, 0) if skipped else (7, 3, 6))),
    )
    EasyMagpieTTSModel._log_phoneme_loss_coverage(model, output)
    assert int(logged['train/phoneme_lexicon_matched_tokens']) == 7
    assert int(logged['train/phoneme_amplified_tokens']) == (0 if skipped else 3)
    assert int(logged['train/phoneme_loss_tokens']) == (0 if skipped else 6)
    assert float(logged['train/phoneme_amplified_percent']) == (0.0 if skipped else 50.0)
    model.cfg['phoneme_lexicon_loss_scale'] = 1.0
    model.log_dict = lambda *args, **kwargs: pytest.fail('Default scale must not log or synchronize coverage')
    EasyMagpieTTSModel._log_phoneme_loss_coverage(model, None)


def _coverage_rank_worker(rank, rendezvous, directory):
    torch.distributed.init_process_group(
        'gloo', init_method='file://' + rendezvous, rank=rank, world_size=2, timeout=datetime.timedelta(seconds=30)
    )
    try:
        captured = {}

        def capture(metrics, **kwargs):
            captured.update({key: float(value) for key, value in metrics.items()})

        model = SimpleNamespace(cfg={'phoneme_lexicon_loss_scale': 2.0}, log_dict=capture)
        counts = (3, 2, 7) if rank == 0 else (5, 0, 0)
        output = SimpleNamespace(loss=torch.tensor(0.0), phoneme_loss_coverage=tuple(torch.tensor(x) for x in counts))
        EasyMagpieTTSModel._log_phoneme_loss_coverage(model, output)
        (Path(directory) / f'rank_{rank}.json').write_text(json.dumps(captured))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.unit
@pytest.mark.skipif(sys.platform == 'win32', reason='Uses fork in the CPU-only container test')
def test_logging_sums_counts_across_real_distributed_ranks(tmp_path):
    torch.multiprocessing.start_processes(
        _coverage_rank_worker,
        args=(str(tmp_path / 'rendezvous'), str(tmp_path)),
        nprocs=2,
        start_method='fork',
        join=True,
    )
    for rank in range(2):
        metrics = json.loads((tmp_path / f'rank_{rank}.json').read_text())
        assert metrics['train/phoneme_lexicon_matched_tokens'] == 8
        assert metrics['train/phoneme_amplified_tokens'] == 2
        assert metrics['train/phoneme_loss_tokens'] == 7
        assert metrics['train/phoneme_amplified_percent'] == pytest.approx(100 * 2 / 7)


@pytest.mark.unit
def test_scale_one_and_validation_do_not_require_masks():
    assert make_phoneme_loss_weights(None, None, 1, None, scale=1.0) is None
    assert make_phoneme_loss_weights(None, None, 1, None, scale=3.0, mode='val') is None
    with pytest.raises(ValueError, match='phoneme_loss_mask'):
        make_phoneme_loss_weights(None, torch.ones(1, 2), 1, None, scale=2.0)


@pytest.mark.unit
@pytest.mark.parametrize('text_only', [True, False])
def test_production_batch_loss_routing_preserves_logits_and_other_losses(text_only):
    from tests.collections.tts.test_easy_magpietts_repeat_context import _batch, _ChannelHarness

    torch.manual_seed(42)
    model = _ChannelHarness()
    model.phoneme_tokenizer = SimpleNamespace(bos_token_id=29, eos_token_id=30, pad=0)
    model.phoneme_corruption_batch_prob = 0.0
    model.phoneme_corruption_timestep_ratio = 0.0
    model.phoneme_corruption_unk_mode_prob = 0.0
    model.embed_phoneme_tokens = MethodType(lambda self, tokens: self.text_lookup(tokens.long()).sum(dim=1), model)
    model.prepare_phoneme_channel_embeddings = MethodType(EasyMagpieTTSModel.prepare_phoneme_channel_embeddings, model)
    model.corrupt_stacked_phoneme_tokens = MethodType(EasyMagpieTTSModel.corrupt_stacked_phoneme_tokens, model)
    batch = _batch()
    batch['mode'] = 'train'
    batch['phoneme_tokens'] = torch.tensor([[29, 2, 3, 30], [29, 3, 30, 0]])
    batch['phoneme_tokens_lens'] = torch.tensor([4, 3])
    batch['phoneme_loss_mask'] = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)
    if text_only:
        batch['task'] = ['text_only', 'text_only']
    baseline = model.process_batch(**batch)
    model.cfg.phoneme_lexicon_loss_scale = 3.0
    weighted = model.process_batch(**batch)
    torch.testing.assert_close(weighted.phoneme_logits, baseline.phoneme_logits, rtol=0, atol=0)
    torch.testing.assert_close(weighted.codebook_loss, baseline.codebook_loss, rtol=0, atol=0)
    weights = make_phoneme_loss_weights(batch['phoneme_loss_mask'], batch['phoneme_tokens'], 1, 29, scale=3.0)
    expected, _ = model.compute_phoneme_loss(
        baseline.phoneme_logits,
        baseline.phoneme_tokens_target,
        baseline.phoneme_tokens_lens_target,
        token_weights=weights,
    )
    torch.testing.assert_close(weighted.phoneme_loss, expected)
    assert weighted.phoneme_loss > baseline.phoneme_loss
    weighted.loss.backward()
    assert torch.isfinite(model.phoneme_final_proj.weight.grad).all()
    batch['mode'] = 'val'
    validation = model.process_batch(**batch)
    torch.testing.assert_close(validation.phoneme_loss, baseline.phoneme_loss)


@pytest.mark.unit
@pytest.mark.parametrize('scale', [0.5, -1.0, float('nan'), float('inf')])
def test_invalid_dataset_scale_fails_early(scale):
    with pytest.raises(ValueError, match='finite and >= 1.0'):
        MagpieTTSLhotseMultiturnDataset(sample_rate=16000, phoneme_lexicon_loss_scale=scale)


@pytest.mark.unit
def test_missing_path_fails_early():
    with pytest.raises(ValueError, match='lexicon_path is required'):
        MagpieTTSLhotseMultiturnDataset(sample_rate=16000, phoneme_lexicon_loss_scale=2.0)


@pytest.mark.unit
@pytest.mark.parametrize('mode,scale', [('train', 1.0), ('test', 3.0)])
def test_disabled_and_validation_collators_do_not_load_lexicon(tokenizer, mode, scale):
    dataset = MagpieTTSLhotseMultiturnDataset(
        sample_rate=16000,
        codec_model_samples_per_frame=160,
        codec_model_input_sample_rate=16000,
        frame_stacking_factor=1,
        dataset_type=mode,
        phoneme_loss_lexicon_path='/does/not/exist.yaml',
        phoneme_lexicon_loss_scale=scale,
    )
    dataset.phoneme_tokenizer = tokenizer
    result = dataset._collate_phoneme_tokens([_cut()])
    assert 'phoneme_loss_mask' not in result
    assert dataset._phoneme_loss_lexicon is None


@pytest.mark.unit
def test_worker_caches_lexicon_and_returns_mask(lexicon_path, tokenizer):
    dataset = MagpieTTSLhotseMultiturnDataset(
        sample_rate=16000,
        codec_model_samples_per_frame=160,
        codec_model_input_sample_rate=16000,
        frame_stacking_factor=1,
        phoneme_loss_lexicon_path=str(lexicon_path),
        phoneme_lexicon_loss_scale=2.0,
    )
    dataset.phoneme_tokenizer = tokenizer
    result = dataset._collate_phoneme_tokens([_cut()])
    assert result['phoneme_loss_mask'].any()
    matcher = dataset._phoneme_loss_lexicon
    dataset._collate_phoneme_tokens([_cut()])
    assert matcher is dataset._phoneme_loss_lexicon


@pytest.mark.unit
def test_restored_artifact_can_disappear_before_spawned_worker_collation(lexicon_path, tokenizer):
    # Model construction eagerly loads the temporary .nemo artifact. A spawned
    # worker receives the compiled lexicon, even after the extraction is cleaned.
    matcher = pickle.loads(pickle.dumps(PhonemeLossLexicon(lexicon_path)))
    lexicon_path.unlink()
    dataset = MagpieTTSLhotseMultiturnDataset(
        sample_rate=16000,
        codec_model_samples_per_frame=160,
        codec_model_input_sample_rate=16000,
        frame_stacking_factor=1,
        phoneme_loss_lexicon_path=str(lexicon_path),
        phoneme_lexicon_loss_scale=2.0,
        phoneme_loss_lexicon=matcher,
    )
    dataset.phoneme_tokenizer = tokenizer
    assert dataset._collate_phoneme_tokens([_cut()])['phoneme_loss_mask'].any()


@pytest.mark.unit
def test_shipped_lexicon_covers_categories_and_required_words():
    path = Path(__file__).parents[3] / 'examples/tts/conf/magpietts/phoneme_loss_lexicon_en.yaml'
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    words = {entry['word'] for entry in data['entries']}
    assert all(set(entry['pronunciations']) == {'en-US'} for entry in data['entries'])
    assert set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') <= words
    assert (
        not {
            'bee',
            'ee',
            'cee',
            'dee',
            'you',
            'why',
            'zed',
            'haitch',
            'model',
            'website',
            'phone',
            'cryptocurrencys',
            'reals',
            'zeroths',
            'a m',
            'p m',
            'A D',
            'B C',
            'B C E',
            'C E',
        }
        & words
    )
    assert {'AM', 'PM', 'AD', 'BC', 'BCE', 'CE', 'email', 'e-mail', 'reais', 'zeroth', 'oh'} <= words
    assert {
        'million',
        'billion',
        'nineteen',
        'nineteenth',
        'twenty',
        'minus',
        'point',
        'dollars',
        'January',
        'UTC',
        'extension',
        'underscore',
        'dot',
        'www',
        'A',
        'Z',
        'kilogram',
    } <= words
    categories = {category for entry in data['entries'] for category in entry['categories']}
    assert {
        'currency',
        'percent',
        'decimal',
        'fractions',
        'dates',
        'time',
        'years',
        'ordinal',
        'phone',
        'urls',
        'email',
        'alphanumeric',
        'roman_numerals',
        'unit_abbreviations',
        'symbols',
    } <= categories
    PhonemeLossLexicon(path)  # schema, unique words, and all pronunciation variants validated


@pytest.mark.unit
def test_coverage_metrics_reach_the_lightning_logger(tmp_path):
    class CoverageModel(LightningModule):
        _log_phoneme_loss_coverage = EasyMagpieTTSModel._log_phoneme_loss_coverage

        def __init__(self):
            super().__init__()
            self.cfg = {'phoneme_lexicon_loss_scale': 2.0}
            self.weight = nn.Parameter(torch.ones(()))

        def training_step(self, batch, batch_idx):
            loss = self.weight.square()
            counts = (8, 5, 11) if batch_idx == 0 else (3, 0, 0)
            self._log_phoneme_loss_coverage(
                SimpleNamespace(loss=loss, phoneme_loss_coverage=tuple(torch.tensor(x) for x in counts))
            )
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    logger = CSVLogger(str(tmp_path), name='coverage')
    trainer = Trainer(
        accelerator='cpu',
        devices=1,
        max_steps=2,
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(CoverageModel(), torch.utils.data.DataLoader(torch.zeros(2, 1), batch_size=1))
    with open(Path(logger.log_dir) / 'metrics.csv', newline='') as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == 2
    assert float(rows[0]['train/phoneme_amplified_tokens']) == 5
    assert float(rows[0]['train/phoneme_loss_tokens']) == 11
    assert float(rows[0]['train/phoneme_amplified_percent']) == pytest.approx(100 * 5 / 11)
    assert float(rows[1]['train/phoneme_lexicon_matched_tokens']) == 3
    assert float(rows[1]['train/phoneme_amplified_tokens']) == 0
    assert float(rows[1]['train/phoneme_amplified_percent']) == 0


@pytest.mark.unit
@pytest.mark.parametrize('scale', [1.0, 2.0])
def test_eager_lhotse_setup_initializes_lexicon_before_subclass_constructor(
    monkeypatch, lexicon_path, tokenizer, scale
):
    # ModelPT invokes this real factory before EasyMagpieTTSModel.__init__ resumes.
    # No lexicon attributes exist yet, just as in fresh training.
    model = SimpleNamespace(
        cfg=OmegaConf.create(
            dict(
                use_multiturn_dataset=True,
                phoneme_lexicon_loss_scale=scale,
                phoneme_loss_lexicon_path=str(lexicon_path),
                load_cached_codes_if_available=False,
                context_duration_min=0.1,
                context_duration_max=1.0,
                text_tokenizers={},
            )
        ),
        phoneme_tokenizer=tokenizer,
        sample_rate=16000,
        codec_model_samples_per_frame=160,
        codec_model_input_sample_rate=16000,
        frame_stacking_factor=1,
        data_num_audio_codebooks=1,
        pad_context_text_to_max_duration=False,
        text_conditioning_tokenizer_name='tiny',
        enable_phoneme_text_input=False,
        text_phoneme_token_offset=0,
        partial_phoneme_text_prob=0.0,
        partial_phoneme_portion_min=0.0,
        partial_phoneme_portion_max=1.0,
        phoneme_text_bop_marker=None,
        phoneme_text_eop_marker=None,
        add_language_to_context_text=False,
        global_rank=0,
        world_size=1,
    )
    artifact_paths = []

    def register_artifact(name, path):
        artifact_paths.append((name, path))
        return path

    model.register_artifact = register_artifact
    model._initialize_phoneme_loss_lexicon = MethodType(EasyMagpieTTSModel._initialize_phoneme_loss_lexicon, model)
    monkeypatch.setattr('nemo.collections.tts.models.easy_magpietts.FallbackDataset', lambda dataset: dataset)
    monkeypatch.setattr(
        'nemo.collections.tts.models.easy_magpietts.get_lhotse_dataloader_from_config',
        lambda **kwargs: kwargs['dataset'],
    )
    config = OmegaConf.create(dict(volume_norm=False, dataset={}))
    dataset = EasyMagpieTTSModel.get_lhotse_dataloader(model, config)
    assert model.phoneme_lexicon_loss_scale == scale
    assert dataset._phoneme_loss_lexicon is model.phoneme_loss_lexicon
    model._initialize_phoneme_loss_lexicon()
    assert len(artifact_paths) == (1 if scale > 1.0 else 0)
    dataset.phoneme_tokenizer = tokenizer
    result = dataset._collate_phoneme_tokens([_cut()])
    if scale > 1.0:
        assert result['phoneme_loss_mask'].any()
    else:
        assert 'phoneme_loss_mask' not in result


@pytest.mark.unit
def test_shipped_canonical_letters_keep_normalization_pronunciations(tokenizer):
    path = Path(__file__).parents[3] / 'examples/tts/conf/magpietts/phoneme_loss_lexicon_en.yaml'
    lexicon = PhonemeLossLexicon(path)
    # B/E remain useful for spelled identifiers without fake written aliases.
    # The final word is ordinary speech with no normalization-vocabulary match.
    ipa = 'bˈiː ˈiː həlˈoʊ'
    ids, matched = lexicon.encode_with_mask(ipa, tokenizer)
    assert ids == tokenizer.encode(ipa)
    assert lexicon.find_spans(ipa) == [(0, 4), (5, 8)]
    assert any(matched) and not all(matched)
    entries = {entry['word']: entry for entry in yaml.safe_load(path.read_text(encoding='utf-8'))['entries']}
    a = entries['A']['pronunciations']['en-US'][0]
    d = entries['D']['pronunciations']['en-US'][0]
    assert entries['AD']['pronunciations']['en-US'] == [a + ' ' + d]
