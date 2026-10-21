# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loader-side word matching must change only text dependencies, never invent IPA boundaries."""

import copy
import hashlib
import json
import re

import pytest
import torch
from lhotse import CutSet
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models, pre_tokenizers

from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config
from nemo.collections.common.data.lhotse.text_adapters import LhotseTextNormJsonlAdapter
from nemo.collections.common.data.lhotse.text_alignment import rebuild_text_word_alignment, resolve_word_alignment
from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import MagpieTTSLhotseMultiturnDataset
from nemo.collections.tts.parts.utils.dynamic_phoneme_delay import phoneme_source_dependencies

pytestmark = pytest.mark.unit


def example(text='Pay 12 then 34 now', normalized='Pay twelve then thirty four now'):
    words = lambda value: [[m.start(), m.end()] for m in re.finditer(r'\S+', value)]
    ipa = ' '.join(f'p{i}' for i in range(len(words(normalized))))
    return dict(
        id='example',
        text=text,
        text_normalized=normalized,
        ipa=ipa,
        num_tokens=len(text),
        word_alignment=dict(
            offset_unit='unicode_code_points',
            ranges='zero_based_end_exclusive',
            string_sha256={
                k: hashlib.sha256(v.encode()).hexdigest()
                for k, v in [('text', text), ('text_normalized', normalized), ('ipa', ipa)]
            },
            source_words=words(text),
            normalized_words=words(normalized),
            normalized_to_ipa=words(ipa),
            source_to_normalized=[None] * len(words(text)),
            groups=[],
            unresolved_regions=[
                dict(source_words=[0, len(words(text))], normalized_words=[0, len(words(normalized))])
            ],
            source_ipa_shared_groups=[],
            normalized_ipa_shared_groups=[],
        ),
    )


def rebuild(row):
    return rebuild_text_word_alignment(row['word_alignment'], row['text'], row['text_normalized'], row['ipa'])


def resolved(row):
    return resolve_word_alignment(rebuild(row), row['text'], row['text_normalized'], row['ipa'])


def test_rebuild_splits_coarse_source_regions_at_unchanged_anchors_without_mutating_input():
    row = example()
    before = copy.deepcopy(row)
    original = resolve_word_alignment(row['word_alignment'], row['text'], row['text_normalized'], row['ipa'])
    assert original['source'] == ((len(row['text']), 0, len(row['ipa'])),)
    groups = resolved(row)['source']
    assert groups == ((3, 0, 2), (6, 3, 5), (11, 6, 8), (14, 9, 14), (18, 15, 17))
    assert row == before


def test_repeated_conjunction_inside_number_is_not_a_greedy_anchor():
    row = example('Pay 101 and 2 now', 'Pay one hundred and one and two now')
    groups = resolved(row)['source']
    assert groups[1] == (13, 3, 20)  # both numbers stay atomic: the two "and" choices are ambiguous
    assert len(groups) == 3


def test_multiple_adjacent_changed_words_are_not_split_by_equal_word_counts():
    row = example('Pay 12 34 now', 'Pay twelve thirty four now')
    assert resolved(row)['source'][1] == (9, 3, 11)


def test_case_punctuation_and_attached_dash_can_anchor_without_changing_offsets():
    row = example('The 1770s—like 12 entries.', 'the seventeen seventies — like twelve entries.')
    a = rebuild(row)
    groups = resolve_word_alignment(a, row['text'], row['text_normalized'], row['ipa'])['source']
    assert any(end == row['text'].index('—') for end, _, _ in groups)
    assert a['normalized_words'] == row['word_alignment']['normalized_words']
    assert a['normalized_to_ipa'] == row['word_alignment']['normalized_to_ipa']


def test_shared_ipa_spans_remain_atomic_after_rebuilding_raw_text():
    row = example('say of the 2 now', 'say of the two now')
    a = row['word_alignment']
    a['normalized_to_ipa'][1:3] = [None, None]
    a['normalized_ipa_shared_groups'] = [dict(normalized_words=[1, 3], ipa_chars=[3, 8])]
    assert resolved(row)['source'][1] == (10, 3, 8)


def test_ipa_boundary_available_only_in_source_shared_metadata_is_preserved():
    row = example()
    a = row['word_alignment']
    a['normalized_to_ipa'][1] = None
    a['source_ipa_shared_groups'] = [dict(source_words=[1, 2], normalized_words=[1, 2], ipa_chars=[3, 5])]
    assert resolved(row) is not None


@pytest.mark.parametrize('kind', ['missing', 'hash', 'ipa_hole', 'explicit_false', 'bad_range'])
def test_rebuild_does_not_bypass_unusable_ipa_metadata(kind):
    row = example()
    if kind == 'missing':
        row['word_alignment'] = None
    elif kind == 'hash':
        row['word_alignment']['string_sha256']['text'] = 'stale'
    elif kind == 'ipa_hole':
        row['word_alignment']['normalized_to_ipa'][1] = None
    elif kind == 'explicit_false':
        row['word_alignment']['dynamic_delay_usable'] = False
    else:
        row['word_alignment']['normalized_words'][0] = [0, 9999]
    assert rebuild(row) is None


def test_insertions_and_deletions_use_bounded_neighbor_context():
    for raw, norm in [('Pay now', 'Pay the fee now'), ('Pay uh now', 'Pay now'), ('now', 'the fee now')]:
        row = example(raw, norm)
        assert resolved(row) is not None


def test_moved_number_cannot_be_treated_as_an_unspoken_deletion():
    row = example('May 1', 'first May')
    assert resolved(row)['source'] == ((5, 0, len(row['ipa'])),)


def test_actual_bpe_offsets_still_control_waits_and_cross_word_ipa_tokens():
    row = example()
    a = rebuild(row)
    offsets = [(i, i + 1) for i in range(len(row['text']))]
    deps, bad = phoneme_source_dependencies(
        a, row['text'], row['text_normalized'], row['ipa'], row['text'], offsets, [(0, len(row['ipa']))]
    )
    assert not bad and deps == {0: len(row['text'])}
    deps, bad = phoneme_source_dependencies(
        a,
        row['text'],
        row['text_normalized'],
        row['ipa'],
        row['text'],
        offsets,
        [(m.start(), m.end()) for m in re.finditer(r'\S+', row['ipa'])],
    )
    assert not bad and deps[1] == 6  # "12" plus its boundary; no extra fixed delay


def adapter_config(tmp_path, row):
    manifest = tmp_path / 'data.jsonl'
    manifest.write_text(json.dumps(row) + '\n')
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / 'tokenizer.json'
    tokenizer.save(str(path))
    return dict(
        paths=str(manifest),
        token_equivalent_duration=0.08,
        dynamic_phoneme_delay=True,
        duration_text_tokenizer_path=str(path),
        duration_phoneme_tokenizer_path=str(path),
    )


def test_adapter_opt_in_reuses_rebuilt_metadata_and_duration_bound(tmp_path):
    row = example()
    cfg = adapter_config(tmp_path, row)
    old = list(LhotseTextNormJsonlAdapter(**cfg))[0]
    new = list(LhotseTextNormJsonlAdapter(**cfg, dynamic_phoneme_delay_recompute_alignment=True))[0]
    assert old.duration == new.duration
    assert len(old.supervisions[0].custom['_dynamic_phoneme_alignment']['source']) == 1
    custom = new.supervisions[0].custom
    assert len(custom['_dynamic_phoneme_alignment']['source']) == 5
    assert custom['word_alignment']['text_alignment_method'] == 'loader_word_anchors_v1'
    assert new.custom['word_alignment'] == custom['word_alignment']
    assert custom['text'] == row['text'] and custom['ipa'] == row['ipa']


def test_recompute_requires_dynamic_mode(tmp_path):
    cfg = adapter_config(tmp_path, example())
    cfg['dynamic_phoneme_delay'] = False
    with pytest.raises(ValueError, match='requires dynamic_phoneme_delay'):
        LhotseTextNormJsonlAdapter(**cfg, dynamic_phoneme_delay_recompute_alignment=True)


def test_nested_input_configuration_propagates_recompute_flag(tmp_path):
    cfg = adapter_config(tmp_path, example())
    path = cfg.pop('paths')
    cfg.update(
        dynamic_phoneme_delay_recompute_alignment=True,
        force_finite=True,
        input_cfg=[
            dict(
                type='group',
                input_cfg=[
                    dict(
                        type='txt_norm_jsonl',
                        paths=path,
                        **{k: v for k, v in cfg.items() if k.startswith('duration_')},
                    )
                ],
            )
        ],
    )
    cuts, _ = read_cutset_from_config(OmegaConf.create(cfg))
    assert (
        next(iter(cuts)).supervisions[0].custom['word_alignment']['text_alignment_method'] == 'loader_word_anchors_v1'
    )


def test_native_collator_consumes_rebuilt_alignment_without_changing_input_or_target_content(tmp_path):
    class CharacterTokenizer:
        pad, bos_token_id, eos_token_id = 0, 10000, 10001
        tokenizer_pad_ids = {'text': 0}

        def encode(self, text, **kwargs):
            return [ord(char) + 1 for char in text]

        def encode_with_offsets(self, text, **kwargs):
            return self.encode(text), [(i, i + 1) for i in range(len(text))]

    row = example()
    cfg = adapter_config(tmp_path, row)
    original = CutSet.from_cuts(list(LhotseTextNormJsonlAdapter(**cfg)))
    rebuilt = CutSet.from_cuts(list(LhotseTextNormJsonlAdapter(**cfg, dynamic_phoneme_delay_recompute_alignment=True)))
    ds = MagpieTTSLhotseMultiturnDataset(
        sample_rate=16000,
        codec_model_samples_per_frame=320,
        codec_model_input_sample_rate=16000,
        frame_stacking_factor=1,
        num_audio_codebooks=8,
        text_conditioning_tokenizer_name='text',
        dynamic_phoneme_delay=True,
        load_normalized_text_percent=0.0,
    )
    ds.text_tokenizer = ds.phoneme_tokenizer = CharacterTokenizer()
    ds.pad_id, ds.bos_id, ds.eos_id, ds.interruption_token_id = 0, 20000, 20001, 20002
    before, after = ds[original], ds[rebuilt]
    assert torch.equal(before['text'], after['text'])
    assert torch.equal(before['source_tokens'], after['source_tokens'])
    assert after['dynamic_phoneme_wait_count'].item() < before['dynamic_phoneme_wait_count'].item()
    assert (
        before['phoneme_tokens'][before['phoneme_tokens'] != 0].tolist()
        == after['phoneme_tokens'][after['phoneme_tokens'] != 0].tolist()
    )
