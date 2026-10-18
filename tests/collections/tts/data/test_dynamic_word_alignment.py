# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordinary-word readiness, exact shared-span recovery and pre-batch filtering."""

import hashlib
import json
import re

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from nemo.collections.common.data.lhotse.text_adapters import LhotseTextNormJsonlAdapter
from nemo.collections.common.data.lhotse.text_alignment import resolve_word_alignment
from nemo.collections.tts.parts.utils.dynamic_phoneme_delay import insert_phoneme_waits, phoneme_source_dependencies

pytestmark = pytest.mark.unit


def sample(text='Ofsted next', normalized=None, ipa='ɒfstɛd nɛkst'):
    normalized = text if normalized is None else normalized
    words = lambda value: [[m.start(), m.end()] for m in re.finditer(r'\S+', value)]
    source, norm, targets = words(text), words(normalized), words(ipa)
    assert len(source) == len(norm) == len(targets)
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
            source_words=source,
            normalized_words=norm,
            source_to_normalized=[[i, i + 1] for i in range(len(source))],
            normalized_to_ipa=targets,
            groups=[
                dict(
                    source_words=[0, len(source)],
                    normalized_words=[0, len(norm)],
                    source_chars=[0, len(text)],
                    normalized_chars=[0, len(normalized)],
                    ipa_chars=[0, len(ipa)],
                    changed=False,
                    mapping='one_to_one',
                    status='ok',
                )
            ],
            source_ipa_shared_groups=[],
            normalized_ipa_shared_groups=[],
            unresolved_regions=[],
        ),
    )


def resolve(row):
    return resolve_word_alignment(row['word_alignment'], row['text'], row['text_normalized'], row['ipa'])


def test_ordinary_word_waits_for_boundary_using_actual_bpe_offsets():
    row = sample()
    offsets = [(0, 2), (2, 6), (6, 11)]  # Of | sted | next (with its leading space)
    deps, unusable = phoneme_source_dependencies(
        row['word_alignment'], row['text'], row['text_normalized'], row['ipa'], row['text'], offsets, [(0, 6), (6, 12)]
    )
    assert not unusable
    assert deps == {0: 2, 1: 3}
    assert insert_phoneme_waits([80, 11, 12, 81], {i + 1: j for i, j in deps.items()}, 0)[0] == [80, 0, 0, 11, 12, 81]


def test_long_unchanged_group_stays_word_level():
    row = sample('first second third', ipa='fɜːst sɛkənd θɜːd')
    groups = resolve(row)['source']
    assert [group[0] for group in groups] == [5, 12, 18]


def test_prior_emission_avoids_unnecessary_per_word_waits():
    # A long number pronunciation leaves later source words already visible.
    tokens, _, waits = insert_phoneme_waits([80] + [11] * 30 + [12, 81], {1: 2, 31: 8}, 0)
    assert waits == 2
    assert tokens[-3:] == [11, 12, 81]


def test_shared_phrase_crossing_normalization_group_is_recovered_locally():
    row = sample('say of the rest', ipa='seɪ əv ðə ɹɛst')
    a = row['word_alignment']
    a['normalized_to_ipa'][1:3] = [None, None]
    a['normalized_ipa_shared_groups'] = [dict(normalized_words=[1, 3], ipa_chars=[4, 9])]
    a['groups'] = [
        dict(
            source_words=[1, 2],
            normalized_words=[1, 2],
            source_chars=[4, 6],
            normalized_chars=[4, 6],
            ipa_chars=None,
            changed=True,
            status='partial',
        )
    ]
    assert resolve(row)['source'] == ((3, 0, 3), (10, 4, 9), (15, 10, 14))
    assert resolve(row)['source'][1][0] < len(row['text'])


def test_shared_span_closure_reaches_source_word_with_normalization_expansion():
    row = sample('say of the rest', ipa='seɪ əv ðə ɹɛst')
    a = row['word_alignment']
    a['source_words'] = [[0, 3], [4, 10], [11, 15]]
    a['source_to_normalized'] = [[0, 1], [1, 3], [3, 4]]
    a['groups'] = []
    a['normalized_to_ipa'][2] = None
    a['normalized_ipa_shared_groups'] = [dict(normalized_words=[2, 4], ipa_chars=[7, 14])]
    raw, norm = resolve(row)['source'], resolve(row)['normalized']
    assert raw == ((3, 0, 3), (15, 4, 14))
    assert norm == ((3, 0, 3), (6, 4, 6), (15, 7, 14))


def test_entity_dependency_remains_atomic_only_for_raw_input():
    row = sample('one hundred next', ipa='wʌn hʌndɹəd nɛkst')
    a = row['word_alignment']
    a['groups'] = [dict(source_words=[0, 2], normalized_words=[0, 2], changed=True, entity_type='number', status='ok')]
    assert resolve(row)['source'][0][0] == 11
    assert resolve(row)['normalized'][0][0] == 3


@pytest.mark.parametrize('kind', ['missing', 'stale', 'ipa_hole', 'missing_boundary', 'explicit_false', 'bad_range'])
def test_unusable_metadata_is_rejected_without_eos_fallback(kind):
    row = sample()
    a = row['word_alignment']
    if kind == 'missing':
        row['word_alignment'] = None
    elif kind == 'stale':
        a['string_sha256']['ipa'] = 'stale'
    elif kind == 'ipa_hole':
        a['normalized_to_ipa'][0] = [1, 6]
    elif kind == 'missing_boundary':
        a['normalized_to_ipa'][0] = None
    elif kind == 'explicit_false':
        a['dynamic_delay_usable'] = False
    else:
        a['groups'][0]['source_chars'] = [0, 1000]
    assert resolve(row) is None
    assert phoneme_source_dependencies(
        row['word_alignment'],
        row['text'],
        row['text_normalized'],
        row['ipa'],
        row['text'],
        [(0, len(row['text']))],
        [(0, len(row['ipa']))],
    ) == ({}, True)


def test_partial_status_and_usable_flag_do_not_override_validation():
    row = sample()
    row['word_alignment'].update(status='partial', dynamic_delay_usable=True)
    assert resolve(row) is not None
    row['word_alignment']['normalized_to_ipa'][0] = None
    assert resolve(row) is None


def test_ipa_bpe_crossing_two_words_uses_later_dependency():
    row = sample()
    deps, bad = phoneme_source_dependencies(
        row['word_alignment'],
        row['text'],
        row['text_normalized'],
        row['ipa'],
        row['text'],
        [(i, i + 1) for i in range(len(row['text']))],
        [(0, len(row['ipa']))],
    )
    assert not bad and deps == {0: len(row['text'])}


def test_byte_fallback_pieces_at_word_boundary_are_all_visible():
    row = sample()
    deps, bad = phoneme_source_dependencies(
        row['word_alignment'],
        row['text'],
        row['text_normalized'],
        row['ipa'],
        row['text'],
        [(0, 2), (2, 6), (6, 7), (6, 7), (7, 11)],
        [(0, 6), (6, 12)],
    )
    assert not bad and deps == {0: 3, 1: 5}


def test_unresolved_internal_text_boundaries_use_explicit_bounded_phrase():
    row = sample()
    a = row['word_alignment']
    a['source_to_normalized'] = [None, None]
    a['groups'] = [dict(source_words=[0, 1], normalized_words=None, status='partial')]
    a['unresolved_regions'] = [dict(source_words=[0, 2], normalized_words=[0, 2])]
    assert resolve(row)['source'] == ((11, 0, 12),)


def test_missing_text_boundaries_are_not_guessed_from_equal_word_counts():
    row = sample()
    a = row['word_alignment']
    a['source_to_normalized'] = [None, None]
    a['groups'][0].update(status='partial', text_status='partial')
    assert resolve(row) is None


def adapter(tmp_path, rows, enabled):
    manifest = tmp_path / 'data.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, 'next': 1}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / 'tokenizer.json'
    tokenizer.save(str(path))
    return LhotseTextNormJsonlAdapter(
        str(manifest),
        token_equivalent_duration=0.08,
        dynamic_phoneme_delay=enabled,
        duration_text_tokenizer_path=str(path),
        duration_phoneme_tokenizer_path=str(path),
    )


def test_adapter_filters_before_batching_and_preserves_disabled_records(tmp_path, caplog):
    good, bad = sample(), sample()
    bad['id'] = 'bad'
    bad['word_alignment']['normalized_to_ipa'][0] = None
    active = list(adapter(tmp_path, [bad, good], True))
    assert [cut.id for cut in active] == ['example']
    assert '_dynamic_phoneme_alignment' in active[0].supervisions[0].custom
    assert 'excluded 1/2 unusable records' in caplog.text
    assert len(list(adapter(tmp_path, [bad, good], False))) == 2


def test_all_rejected_adapter_raises_instead_of_repeating_empty_input(tmp_path):
    bad = sample()
    bad['word_alignment'] = None
    with pytest.raises(ValueError, match='No usable word alignments'):
        list(adapter(tmp_path, [bad], True))


def test_unspoken_alignment_placeholder_needs_no_source_dependency():
    row = sample()
    a = row['word_alignment']
    a['normalized_words'] = [[0, 6], [6, 7], [7, 11]]
    a['normalized_to_ipa'] = [[0, 6], [6, 7], [7, 12]]
    a['source_to_normalized'] = [[0, 1], [2, 3]]
    a['groups'] = []
    assert resolve(row)['source'] == ((6, 0, 6), (11, 7, 12))
