# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Causal timing, metadata fallback and collation contracts for text-only waits."""

import hashlib
import json

import pytest
import torch
from lhotse import CutSet

from nemo.collections.common.data.lhotse.text_adapters import LhotseTextNormJsonlAdapter
from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import MagpieTTSLhotseMultiturnDataset
from nemo.collections.tts.parts.utils.dynamic_phoneme_delay import insert_phoneme_waits, phoneme_source_dependencies

pytestmark = pytest.mark.unit


class CharacterTokenizer:
    pad = 0
    bos_token_id = 10000
    eos_token_id = 10001
    tokenizer_pad_ids = {'text': 0}

    def encode(self, text, **kwargs):
        return [ord(c) + 1 for c in text]

    def encode_with_offsets(self, text, **kwargs):
        return self.encode(text), [(i, i + 1) for i in range(len(text))]


def row(text='$12345 next', normalized='one next', ipa='wʌn nɛkst'):
    first, nfirst, ifirst = text.split()[0], normalized.split()[0], ipa.split()[0]
    alignment = {
        'version': '1.2.0',
        'offset_unit': 'unicode_code_points',
        'ranges': 'zero_based_end_exclusive',
        'source_words': [[0, len(first)], [len(first) + 1, len(text)]],
        'normalized_words': [[0, len(nfirst)], [len(nfirst) + 1, len(normalized)]],
        'normalized_to_ipa': [[0, len(ifirst)], [len(ifirst) + 1, len(ipa)]],
        'source_ipa_shared_groups': [],
        'normalized_ipa_shared_groups': [],
        'unresolved_regions': [],
        'groups': [
            {
                'source_words': [0, 1],
                'normalized_words': [0, 1],
                'source_chars': [0, len(first)],
                'normalized_chars': [0, len(nfirst)],
                'ipa_chars': [0, len(ifirst)],
                'changed': True,
                'entity_type': 'money',
                'status': 'ok',
            }
        ],
        'string_sha256': {
            k: hashlib.sha256(v.encode()).hexdigest()
            for k, v in {'text': text, 'text_normalized': normalized, 'ipa': ipa}.items()
        },
    }
    return {
        'id': 'fixture',
        'text': text,
        'text_normalized': normalized,
        'ipa': ipa,
        'num_tokens': len(text),
        'language_id': 'en',
        'word_alignment': alignment,
    }


def dependencies(record, selected=None, text_offsets=None, ipa_offsets=None):
    tok = CharacterTokenizer()
    selected = record['text'] if selected is None else selected
    return phoneme_source_dependencies(
        record['word_alignment'],
        record['text'],
        record['text_normalized'],
        record['ipa'],
        selected,
        tok.encode_with_offsets(selected)[1] if text_offsets is None else text_offsets,
        tok.encode_with_offsets(record['ipa'])[1] if ipa_offsets is None else ipa_offsets,
    )


def dataset(enabled, normalized=0.0):
    ds = MagpieTTSLhotseMultiturnDataset(
        sample_rate=16000,
        codec_model_samples_per_frame=320,
        codec_model_input_sample_rate=16000,
        frame_stacking_factor=1,
        num_audio_codebooks=8,
        text_conditioning_tokenizer_name='text',
        dynamic_phoneme_delay=enabled,
        load_normalized_text_percent=normalized,
    )
    ds.text_tokenizer = ds.phoneme_tokenizer = CharacterTokenizer()
    ds.pad_id, ds.bos_id, ds.eos_id, ds.interruption_token_id = 0, 20000, 20001, 20002
    return ds


def cuts(tmp_path, records):
    path = tmp_path / 'samples.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    return CutSet.from_cuts(list(LhotseTextNormJsonlAdapter(str(path), token_equivalent_duration=0.04)))


def test_word_closure_and_eos_are_visible_before_first_phoneme():
    deps, fallback = dependencies(row())
    assert deps == {0: 6} and not fallback  # includes the following space
    r = row(text='$12345', normalized='one', ipa='wʌn')
    assert dependencies(r) == ({0: 6}, False)  # text EOS, not merely the last digit


def test_waits_use_prediction_clock_and_accumulate_without_double_counting():
    ids, mask, count = insert_phoneme_waits([80, 11, 12, 81], {1: 6, 2: 8}, 0, [False, True, True, False])
    assert ids == [80] + [0] * 6 + [11, 0, 12, 81]
    assert mask == [False] * 7 + [True, False, True, False]
    assert count == 7
    assert insert_phoneme_waits([80, 11, 12, 81], {1: 0}, 0)[0] == [80, 11, 12, 81]
    assert insert_phoneme_waits([80] + [11] * 20 + [81], {20: 10}, 0)[2] == 0


def test_raw_whitespace_timestamp_projection_and_normalized_input():
    r = row(text='$12345\n\nnext')
    assert dependencies(r, selected='$12345 next') == ({0: 6}, False)
    assert dependencies(r, selected=r['text_normalized']) == ({0: 3}, False)
    r = row(text='$12345<|24|> next')
    assert dependencies(r, selected='$12345 next') == ({0: 6}, False)


@pytest.mark.parametrize('kind', ['missing', 'stale', 'range', 'transformed'])
def test_unusable_alignment_waits_for_full_text_without_guessing(kind):
    r = row()
    selected = None
    if kind == 'missing':
        r['word_alignment'] = None
    elif kind == 'stale':
        r['word_alignment']['string_sha256']['ipa'] = 'wrong'
    elif kind == 'range':
        r['word_alignment']['groups'][0]['source_chars'] = [0, 999]
    else:
        selected = 'unexpected replacement'
    value, fallback = dependencies(r, selected)
    assert fallback and value == {0: len(selected or r['text'])}


def test_partial_sample_spelling_group_and_cross_boundary_bpe_remain_usable():
    r = row()
    r['word_alignment']['status'] = 'partial'
    r['word_alignment']['groups'][0]['entity_type'] = None
    assert dependencies(r) == ({0: 6}, False)
    r['word_alignment']['groups'][0]['ipa_chars'] = [1, 3]
    # First token covers both the preceding IPA and the start of the expression.
    assert dependencies(r, ipa_offsets=[(0, 2), (2, 4), (4, 9)]) == ({0: 6}, False)


def test_bounded_unresolved_region_recovers_a_coarse_mapping():
    r = row()
    a = r['word_alignment']
    a['unresolved_regions'] = [{'source_words': [0, 1], 'normalized_words': [0, 1]}]
    a['groups'][0].update(status='partial', normalized_chars=None, ipa_chars=None)
    assert dependencies(r) == ({0: 6}, False)


def test_loader_default_and_enabled_paths_preserve_text_and_shift_only_targets(tmp_path, monkeypatch):
    original = row()
    sample = cuts(tmp_path, [original])
    legacy = dataset(False)
    monkeypatch.setattr(
        legacy, '_apply_dynamic_phoneme_delay', lambda *args: pytest.fail('disabled path scheduled waits')
    )
    baseline = legacy[sample]
    assert 'dynamic_phoneme_wait_count' not in baseline
    assert baseline['phoneme_tokens'][0].tolist() == [10000] + CharacterTokenizer().encode(original['ipa']) + [10001]
    active = dataset(True)[cuts(tmp_path, [original])]
    assert torch.equal(active['text'], baseline['text'])
    assert torch.equal(active['source_tokens'], baseline['source_tokens'])
    assert active['dynamic_phoneme_wait_count'].tolist() == [6]
    assert active['dynamic_phoneme_fallback_count'].tolist() == [0]
    assert active['phoneme_tokens'][0, :8].tolist() == [10000] + [0] * 6 + [ord('w') + 1]
    assert active['phoneme_tokens_lens'].item() == baseline['phoneme_tokens_lens'].item() + 6


def test_loader_uses_the_selected_normalized_transcript_and_text_bos(tmp_path):
    ds = dataset(True, normalized=1.0)
    ds.add_text_bos = True
    value = ds[cuts(tmp_path, [row()])]
    assert value['dynamic_phoneme_wait_count'].tolist() == [4]
    assert value['text'][0, :4].tolist() == [ds.bos_id] + CharacterTokenizer().encode('one')


def test_loader_remaps_lexicon_mask_and_batch_lengths(tmp_path):
    ds = dataset(True)
    r = row()
    sample = cuts(tmp_path, [r])
    text = ds._collate_text_channels(sample, ['text'])
    phonemes = ds._collate_phoneme_tokens(sample)
    old_mask = torch.ones_like(phonemes['target_phoneme_tokens'], dtype=torch.bool)
    phonemes['phoneme_loss_mask'] = old_mask
    ds._apply_dynamic_phoneme_delay(sample, ['text'], text, phonemes)
    assert phonemes['phoneme_loss_mask'][0].tolist() == [True] + [False] * 6 + [True] * (old_mask.size(1) - 1)


@pytest.mark.parametrize('enabled,configured_delay', [(True, 0), (True, 8), (True, 15), (False, 8)])
def test_training_dynamic_mode_removes_only_the_fixed_offset(enabled, configured_delay):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel

    class TimingCaptured(Exception):
        pass

    context_lens = torch.tensor([2])
    fake = SimpleNamespace(
        cfg={'dynamic_phoneme_delay': enabled, 'use_multiturn_dataset': True},
        phoneme_tokenizer=SimpleNamespace(eos_token_id=81),
        task_embedding=None,
        disable_cas_for_context_text=False,
        pad_id=0,
        embed_text_tokens=Mock(return_value=torch.zeros(1, 2, 4)),
        prepare_text_channel_embeddings=Mock(return_value=(torch.zeros(1, 6, 4), torch.tensor([6]))),
        prepare_phoneme_channel_embeddings=Mock(side_effect=TimingCaptured),
    )
    with pytest.raises(TimingCaptured):
        EasyMagpieTTSModel._process_text_only_batch(
            fake,
            torch.tensor([[11, 12, 13, 14]]),
            torch.tensor([4]),
            torch.tensor([[5, 6]]),
            context_lens,
            torch.tensor([[80, 7, 8, 81]]),
            torch.tensor([4]),
            training_mode=SimpleNamespace(streaming_phonemes_delay=configured_delay),
        )
    delay = fake.prepare_phoneme_channel_embeddings.call_args.args[2]
    assert delay.tolist() == [2 if enabled else 2 + configured_delay]


def test_audio_batches_never_run_the_wait_scheduler(monkeypatch):
    ds = dataset(True)
    ds._initialize_tokenizers = lambda: None
    ds._prepare_cuts = lambda cuts: (cuts, ['text'])
    ds._collate_audio_channels = lambda cuts: {'target_audio_lens': torch.tensor([10])}
    ds._collate_text_channels = lambda *args: {}
    ds._collate_phoneme_tokens = lambda *args: {}
    ds._collect_cut_features = lambda *args: {}
    ds._maybe_shuffle_context_audio = lambda *args: None
    ds._build_batch_dict = lambda *args: {'audio': 'unchanged'}
    monkeypatch.setattr(ds, '_apply_dynamic_phoneme_delay', lambda *args: pytest.fail('audio timing changed'))
    from types import SimpleNamespace

    assert ds[[SimpleNamespace(task='tts')]] == {'audio': 'unchanged'}


def test_dynamic_duration_requires_real_tokenizer_lengths(tmp_path):
    path = tmp_path / 'unused.jsonl'
    with pytest.raises(ValueError, match='duration_text_tokenizer_path'):
        LhotseTextNormJsonlAdapter(str(path), token_equivalent_duration=0.04, dynamic_phoneme_delay=True)


def test_dynamic_duration_reserves_source_plus_ipa_without_affecting_default(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    tokenizer = Tokenizer(WordLevel({'[UNK]': 0, '$12345': 1, 'next': 2, 'one': 3}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer_path = tmp_path / 'tokenizer.json'
    tokenizer.save(str(tokenizer_path))
    path = tmp_path / 'samples.jsonl'
    path.write_text(json.dumps(row()) + '\n')
    kwargs = dict(
        paths=str(path),
        token_equivalent_duration=0.04,
        duration_text_tokenizer_path=str(tokenizer_path),
        duration_phoneme_tokenizer_path=str(tokenizer_path),
    )
    baseline = next(iter(LhotseTextNormJsonlAdapter(**kwargs)))
    active = next(iter(LhotseTextNormJsonlAdapter(**kwargs, dynamic_phoneme_delay=True)))
    assert baseline.sampling_num_tokens == 11
    assert active.sampling_num_tokens == 11 + 2 + 4
