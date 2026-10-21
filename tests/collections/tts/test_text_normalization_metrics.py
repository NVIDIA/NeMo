# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference-only group selection and full-pipeline text normalization regression tests."""

import json

import pytest
from scripts.tts.evaluate_normalization import compare_models, load_model_records

from nemo.collections.tts.parts.utils.text_normalization_metrics import (
    aggregate_text_scores,
    build_lexicon_index,
    build_normalization_groups,
    edit_alignment,
    score_text_normalization,
)


@pytest.fixture
def lexicon():
    words = [
        'one',
        'two',
        'three',
        'four',
        'five',
        'twenty',
        'hundred',
        'million',
        'dollars',
        'and',
        'a',
        'half',
        'dot',
        'percent',
        'A',
        'I',
    ]
    return build_lexicon_index([{'word': word, 'categories': ['test']} for word in words])


@pytest.mark.unit
def test_numeric_error_is_not_diluted_by_correct_sentence_words(lexicon):
    raw = 'Your balance is $125. Please contact support for details.'
    reference = 'Your balance is one hundred twenty five dollars. Please contact support for details.'
    hypothesis = 'Your balance is one hundred twenty four dollars. Please contact support for details.'
    score = score_text_normalization(raw, reference, hypothesis, lexicon)
    assert score['group_count'] == 1
    assert score['group_exact'] == 0.0
    assert 0 < score['cer'] < 0.1
    assert score['normalization_groups'][0]['hypothesis_text'] == 'one hundred twenty four dollars'


@pytest.mark.unit
def test_group_selection_does_not_depend_on_predictions(lexicon):
    a = score_text_normalization('Pay $2.', 'Pay two dollars.', 'Pay two dollars.', lexicon)
    b = score_text_normalization('Pay $2.', 'Pay two dollars.', '', lexicon)
    assert a['normalization_groups'][0]['reference_text'] == b['normalization_groups'][0]['reference_text']
    assert a['group_exact'] == 1.0 and b['group_exact'] == 0.0
    assert b['cer'] == b['wer'] == 1.0


@pytest.mark.unit
def test_punctuation_breaks_reference_groups_and_ordinary_words_do_not_create_groups(lexicon):
    groups, _ = build_normalization_groups('1, 2.', 'one, two.', lexicon)
    assert [group['reference_words'] for group in groups] == [['one'], ['two']]
    groups, _ = build_normalization_groups('I have one idea.', 'I have one idea.', lexicon)
    assert groups == []


@pytest.mark.unit
def test_hyphen_and_single_connector_follow_reference_grouping(lexicon):
    groups, _ = build_normalization_groups('125', 'one hundred and twenty-five', lexicon)
    assert len(groups) == 1
    assert groups[0]['reference_words'] == ['one', 'hundred', 'and', 'twenty', 'five']


@pytest.mark.unit
@pytest.mark.parametrize('hypothesis', ['one hundred five', 'one hundred and four'])
def test_no_global_removal_of_and_or_acceptance_of_alternative_number_words(lexicon, hypothesis):
    score = score_text_normalization('105', 'one hundred and five', hypothesis, lexicon)
    assert score['group_exact'] == 0.0


@pytest.mark.unit
def test_insertions_at_group_boundary_are_charged_once(lexicon):
    score = score_text_normalization('1, 2', 'one, two', 'one extra two', lexicon)
    groups = score['normalization_groups']
    assert sum(group['insertions'] for group in groups) == 1
    assert groups[0]['exact'] and not groups[1]['exact']


@pytest.mark.unit
def test_exact_matching_is_case_punctuation_and_whitespace_insensitive(lexicon):
    score = score_text_normalization('$2', 'TWO dollars.', 'two   DOLLARS!', lexicon)
    assert score['cer'] == score['wer'] == 0.0
    assert score['group_exact'] == 1.0


@pytest.mark.unit
def test_group_accuracy_is_macro_averaged_over_eligible_turns(lexicon):
    first = score_text_normalization('1, 2', 'one, two', 'one three', lexicon)
    second = score_text_normalization('3', 'three', 'three', lexicon)
    ordinary = score_text_normalization('Hello.', 'Hello.', 'Hello.', lexicon)
    result = aggregate_text_scores([first, second, ordinary])
    assert result['turns'] == 3 and result['eligible_turns'] == 2
    assert result['normalization_group_exact'] == 0.75
    assert result['pooled_group_exact'] == pytest.approx(2 / 3)


@pytest.mark.unit
def test_changed_words_outside_lexicon_are_reported(lexicon):
    _, coverage = build_normalization_groups('example.com', 'example dot com', lexicon)
    assert coverage['changed_words'] > 0
    assert coverage['selected_changed_words'] > 0


@pytest.mark.unit
def test_hallucination_cer_is_not_clipped(lexicon):
    score = score_text_normalization('1', 'one', 'one one one one one', lexicon)
    assert score['cer'] > 1.0 and score['wer'] > 1.0
    assert score['group_exact'] == 0.0


@pytest.mark.unit
def test_alignment_reproducible_with_repeated_words():
    assert edit_alignment(['one', 'one'], ['one']) == [('delete', 0, 0), ('equal', 1, 0)]


@pytest.mark.unit
def test_multiturn_rows_are_joined_by_audio_then_source_turn_not_error_sorted_order(tmp_path):
    source = [{'cut_id': 'conversation', 'text': ['$1', '$2'], 'normalized_text': ['one dollar', 'two dollars']}]
    metrics = tmp_path / 'metrics.json'
    sidecar = tmp_path / 'multiturn_user_audio_turn_manifest.jsonl'
    sidecar.write_text(
        ''.join(
            json.dumps(
                dict(audio_filepath=f'target_audio_{i}.wav', source_sample_idx=0, turn_id=i, text=source[0]['text'][i])
            )
            + '\n'
            for i in range(2)
        )
    )
    metrics.write_text(
        json.dumps(
            [
                dict(
                    gt_audio_filepath=str(tmp_path / 'target_audio_1.wav'),
                    pred_text='two dollars',
                    gt_text='wrong reference',
                ),
                dict(
                    gt_audio_filepath=str(tmp_path / 'target_audio_0.wav'),
                    pred_text='one dollar',
                    gt_text='wrong reference',
                ),
            ]
        )
    )
    records, _ = load_model_records({'metrics': str(metrics)}, tmp_path, source)
    assert records['conversation-turn_000_agent']['reference_text'] == 'one dollar'
    assert records['conversation-turn_001_agent']['hypothesis_text'] == 'two dollars'


@pytest.mark.unit
def test_source_raw_text_mismatch_is_rejected(tmp_path):
    sidecar = tmp_path / 'multiturn_user_audio_turn_manifest.jsonl'
    sidecar.write_text(json.dumps(dict(audio_filepath='target.wav', source_sample_idx=0, turn_id=0, text='$9')) + '\n')
    metrics = tmp_path / 'metrics.json'
    metrics.write_text(json.dumps([dict(gt_audio_filepath=str(tmp_path / 'target.wav'), pred_text='one')]))
    with pytest.raises(ValueError, match='raw reference mismatch'):
        load_model_records(
            {'metrics': str(metrics)}, tmp_path, [{'cut_id': 'c', 'text': ['$1'], 'normalized_text': ['one']}]
        )


@pytest.mark.unit
def test_common_reference_replaces_inconsistent_stored_references(lexicon):
    record = dict(id='c', raw_text='$1', reference_text='one', hypothesis_text='one')
    models = {'a': {'c': dict(record, stored_reference='twenty')}, 'b': {'c': dict(record, stored_reference='one')}}
    summary, _ = compare_models(models, lexicon)
    assert summary['models']['a']['cer'] == summary['models']['b']['cer'] == 0.0
    assert summary['models']['a']['stored_reference_word_mismatches'] == 1


@pytest.mark.unit
@pytest.mark.parametrize('selected', ['$1', 'one dollar'])
def test_recorded_model_input_may_be_raw_or_normalized(tmp_path, selected):
    metrics = tmp_path / 'metrics.json'
    metrics.write_text(
        json.dumps(
            [
                dict(
                    id='one',
                    gt_audio_filepath=str(tmp_path / 'target.wav'),
                    pred_text='one dollar',
                    tts_text_input=selected,
                )
            ]
        )
    )
    records, _ = load_model_records(
        {'metrics': str(metrics)}, tmp_path, [dict(id='one', text='$1', normalized_text='one dollar')]
    )
    assert records['one']['model_input_text'] == selected
    assert records['one']['raw_text'] == '$1'
    assert records['one']['reference_text'] == 'one dollar'


@pytest.mark.unit
def test_recorded_model_input_from_a_different_sample_is_rejected(tmp_path):
    metrics = tmp_path / 'metrics.json'
    metrics.write_text(
        json.dumps(
            [
                dict(
                    id='one',
                    gt_audio_filepath=str(tmp_path / 'target.wav'),
                    pred_text='one dollar',
                    tts_text_input='two dollars',
                )
            ]
        )
    )
    with pytest.raises(ValueError, match='saved TTS input differs'):
        load_model_records(
            {'metrics': str(metrics)}, tmp_path, [dict(id='one', text='$1', normalized_text='one dollar')]
        )


@pytest.mark.unit
@pytest.mark.parametrize('multiturn', [False, True])
@pytest.mark.parametrize('bad_field', [None, 'original_text', 'tts_text_input', 'phoneme_source'])
def test_frontend_ipa_text_input_keeps_raw_and_normalized_metric_references(tmp_path, multiturn, bad_field):
    ipa = '<bop>wʌn<eop>'
    sidecar = dict(
        audio_filepath='target.wav',
        text=ipa,
        original_text='$1',
        normalized_text='one dollar',
        phoneme_frontend_as_text_input=True,
        phoneme_source='text_frontend',
        frontend_phoneme_text='wʌn',
    )
    sidecar.update(dict(source_sample_idx=0, turn_id=0) if multiturn else dict(item_idx=0))
    metrics = dict(gt_audio_filepath=str(tmp_path / 'target.wav'), pred_text='one dollar', tts_text_input=ipa)
    if bad_field == 'tts_text_input':
        metrics[bad_field] = '<bop>tu<eop>'
    elif bad_field is not None:
        sidecar[bad_field] = 'incorrect'
    manifest = tmp_path / 'turn_manifest.jsonl'
    manifest.write_text(json.dumps(sidecar) + '\n')
    path = tmp_path / 'metrics.json'
    path.write_text(json.dumps([metrics]))
    source = [
        dict(
            id='one', text=['$1'] if multiturn else '$1', normalized_text=['one dollar'] if multiturn else 'one dollar'
        )
    ]
    spec = dict(metrics=str(path), turn_manifest=str(manifest))
    if bad_field:
        with pytest.raises(ValueError, match='mismatch|differs|provenance'):
            load_model_records(spec, tmp_path, source)
    else:
        records, _ = load_model_records(spec, tmp_path, source)
        entry = next(iter(records.values()))
        assert entry['model_input_text'] == ipa
        assert entry['raw_text'] == '$1' and entry['reference_text'] == 'one dollar'


@pytest.mark.unit
def test_different_cohorts_require_explicit_intersection(lexicon):
    record = dict(id='c', raw_text='$1', reference_text='one', hypothesis_text='one')
    models = {'a': {'c': record, 'd': dict(record, id='d')}, 'b': {'c': record}}
    with pytest.raises(ValueError, match='coverage differs'):
        compare_models(models, lexicon)
    summary, _ = compare_models(models, lexicon, cohort='intersection')
    assert summary['common_turns'] == 1
    assert summary['models']['a']['excluded_turn_ids'] == ['d']


@pytest.mark.unit
def test_mismatched_common_reference_is_not_silently_used(lexicon):
    a = dict(id='c', raw_text='$1', reference_text='one', hypothesis_text='one')
    b = dict(a, reference_text='nine')
    with pytest.raises(ValueError, match='Inconsistent raw/reference'):
        compare_models({'a': {'c': a}, 'b': {'c': b}}, lexicon)


@pytest.mark.unit
@pytest.mark.parametrize('field,value', [('source_sample_idx', -1), ('source_sample_idx', True), ('turn_id', -1)])
def test_invalid_sidecar_indices_cannot_wrap_to_last_source_record(tmp_path, field, value):
    row = dict(audio_filepath='target.wav', source_sample_idx=0, turn_id=0, text='$1')
    row[field] = value
    (tmp_path / 'multiturn_user_audio_turn_manifest.jsonl').write_text(json.dumps(row) + '\n')
    metrics = tmp_path / 'metrics.json'
    metrics.write_text(json.dumps([dict(gt_audio_filepath=str(tmp_path / 'target.wav'), pred_text='one')]))
    with pytest.raises(ValueError, match='invalid'):
        load_model_records(
            {'metrics': str(metrics)}, tmp_path, [{'cut_id': 'c', 'text': ['$1'], 'normalized_text': ['one']}]
        )
