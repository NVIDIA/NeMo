# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a TTS/ASR semantic judge response. No model calls or semantic grading.
Structure, cross-field consistency and literal evidence are checked with stdlib.
"""
import json
import sys
from pathlib import Path

TOP_KEYS = {
    'score_1_to_5',
    'verdict',
    'checks',
    'categories_checked',
    'meaning_errors',
    'review_reasons',
    'wording_notes',
    'observability_limits',
    'error_attribution',
    'audio_assessed',
    'summary',
}
CHECK_KEYS = {'values_and_entities', 'claims_and_relationships', 'content_coverage'}
ERROR_TYPES = {
    'minor_nuance_changed',
    'value_changed',
    'unit_or_currency_changed',
    'identifier_changed',
    'date_or_time_changed',
    'negation_or_modality_changed',
    'entity_or_relationship_changed',
    'omission',
    'addition',
    'wrong_contextual_interpretation',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_judgment(judgment, pair):
    require(
        isinstance(judgment, dict) and set(judgment) == TOP_KEYS,
        'Response must contain exactly the required top-level keys.',
    )
    require(
        isinstance(pair, dict) and set(pair) == {'source_text', 'asr_transcript'},
        'Input must contain exactly source_text and asr_transcript; no config is required.',
    )
    source = pair['source_text']
    transcript = pair['asr_transcript']
    require(isinstance(source, str) and isinstance(transcript, str), 'Both input texts must be strings.')
    checks = judgment['checks']
    require(isinstance(checks, dict) and set(checks) == CHECK_KEYS, 'Invalid check keys.')
    active = []
    for key, value in checks.items():
        allowed = ('PASS', 'FAIL', 'REVIEW')
        if key != 'content_coverage':
            allowed += ('NOT_APPLICABLE',)
        require(value in allowed, f'Invalid check value for {key}.')
        if value != 'NOT_APPLICABLE':
            active.append(value)
    for field in ('categories_checked', 'review_reasons', 'wording_notes', 'observability_limits'):
        values = judgment[field]
        require(
            isinstance(values, list) and all(isinstance(s, str) and s.strip() for s in values),
            f'{field} must be a list of nonempty strings.',
        )
    categories = judgment['categories_checked']
    require(len(categories) == len(set(categories)), 'Duplicate category names.')
    errors = judgment['meaning_errors']
    require(isinstance(errors, list), 'meaning_errors must be a list.')
    for error in errors:
        require(
            isinstance(error, dict) and set(error) == {'type', 'source_evidence', 'asr_evidence', 'explanation'},
            'Invalid error keys.',
        )
        require(isinstance(error['type'], str) and error['type'] in ERROR_TYPES, 'Unknown error type.')
        require(
            isinstance(error['explanation'], str) and error['explanation'].strip(),
            'Each error needs a specific explanation.',
        )
        evidence_count = 0
        for field, text in [('source_evidence', source), ('asr_evidence', transcript)]:
            evidence = error[field]
            require(
                evidence is None or (isinstance(evidence, str) and bool(evidence) and evidence in text),
                f'{field} must be a nonempty literal substring or null.',
            )
            evidence_count += evidence is not None
        require(evidence_count > 0, 'An error needs evidence from at least one text.')
    expected = 'FAIL' if 'FAIL' in active else 'REVIEW' if 'REVIEW' in active else 'PASS'
    require(judgment['verdict'] == expected, 'Verdict contradicts applicable checks.')
    if expected == 'FAIL':
        require(bool(errors), 'FAIL requires an evidence-supported meaning error.')
    elif expected == 'REVIEW':
        require(
            not errors and bool(judgment['review_reasons']),
            'REVIEW requires material uncertainty, without a definite meaning error.',
        )
    else:
        require(
            not errors and not judgment['review_reasons'],
            'PASS cannot contain meaning errors or material unresolved ambiguity.',
        )
    score = judgment['score_1_to_5']
    if expected == 'REVIEW':
        require(score is None, 'REVIEW must have a null score, not a fabricated zero.')
    else:
        require(type(score) is int and 1 <= score <= 5, 'Score must be an integer from 1 to 5.')
        require((score == 5) == (expected == 'PASS'), 'Only a fully preserved PASS can score 5.')
        if score == 4:
            require(
                all(error['type'] == 'minor_nuance_changed' for error in errors),
                'Score 4 cannot contain a critical or substantive error type.',
            )
    require(
        judgment['error_attribution'] == 'UNDETERMINED_FROM_TEXT',
        'Text alone cannot attribute a mismatch to TTS versus ASR.',
    )
    require(judgment['audio_assessed'] is False, 'No audio was assessed by this text judge.')
    require(isinstance(judgment['summary'], str) and judgment['summary'].strip(), 'Summary is required.')


def derive_scores(judgment):
    """Deterministic scaling of a validated judgment; not a confidence estimate."""
    score = judgment['score_1_to_5']
    if judgment['verdict'] == 'REVIEW':
        require(score is None, 'REVIEW requires a null score.')
        return {'fully_correct': None, 'semantic_score_0_to_1': None}
    require(type(score) is int and 1 <= score <= 5, 'Invalid ordinal score.')
    require(judgment['verdict'] == ('PASS' if score == 5 else 'FAIL'), 'Ordinal score and verdict disagree.')
    return {'fully_correct': int(score == 5), 'semantic_score_0_to_1': (score - 1) / 4}


def aggregate_scores(judgments):
    """Aggregate validated responses; None marks an unresolved infrastructure record.
    Caller must validate against the matched source/transcript first and ensure one
    entry per predeclared evaluation record. This function cannot verify alignment.
    """
    rows = list(judgments)
    hist = {str(score): 0 for score in range(1, 6)}
    review = missing = 0
    for row in rows:
        if row is None:
            missing += 1
            continue
        derived = derive_scores(row)
        if derived['fully_correct'] is None:
            review += 1
        else:
            hist[str(row['score_1_to_5'])] += 1
    total = len(rows)
    scored = sum(hist.values())
    passed = hist['5']
    failed = scored - passed
    unresolved = review + missing
    score_sum = sum(int(k) * v for k, v in hist.items())
    normalized_sum = sum(((int(k) - 1) / 4) * v for k, v in hist.items())
    return {
        'total': total,
        'fully_correct': passed,
        'incorrect': failed,
        'judge_review': review,
        'missing_result': missing,
        'unresolved': unresolved,
        'score_histogram': hist,
        'fully_correct_rate': passed / total if total else None,
        'unresolved_rate': unresolved / total if total else None,
        'scored_coverage': scored / total if total else None,
        'mean_score_1_to_5_on_scored': score_sum / scored if scored else None,
        'mean_score_0_to_1_on_scored': normalized_sum / scored if scored else None,
        'all_case_mean_0_to_1_missing_score_bounds': (
            [normalized_sum / total, (normalized_sum + unresolved) / total] if total else None
        ),
    }


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate JSON key: {key}.')
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=reject_duplicate_keys)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Usage: python validate_judgment.py INPUT_PAIR.json JUDGMENT.json')
    input_pair, response = read_json(sys.argv[1]), read_json(sys.argv[2])
    validate_judgment(response, input_pair)
    print(
        json.dumps(
            {
                'contract_valid': True,
                'verdict': response['verdict'],
                'score_1_to_5': response['score_1_to_5'],
                **derive_scores(response),
            }
        )
    )
