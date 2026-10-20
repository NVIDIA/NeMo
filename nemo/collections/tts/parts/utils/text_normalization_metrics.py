# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic reference-selected normalization-group scores for ASR transcripts.

No model, phonemizer, or text-normalization engine is invoked. Group selection uses
only raw/reference text and the orthographic words of the phoneme loss lexicon.
"""

import difflib
import re
import statistics
import unicodedata
from collections import Counter, defaultdict

WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
GAP = re.compile(r'[\s\-–—]*')


def tokenize_words(text):
    """Return word strings, canonical keys, and offsets; case/punctuation are not scored."""
    text = unicodedata.normalize('NFKC', text)
    return [
        dict(word=m.group(), key=m.group().replace('’', "'").casefold(), start=m.start(), end=m.end())
        for m in WORD.finditer(text)
    ]


def build_lexicon_index(entries):
    """Index the YAML's words/phrases without loading or using their IPA pronunciations."""
    index = defaultdict(list)
    for entry in entries:
        keys = [word['key'] for word in tokenize_words(entry['word'])]
        if keys:
            index[keys[0]].append(dict(word=entry['word'], keys=keys, categories=entry.get('categories', [])))
    return {key: sorted(values, key=lambda item: (-len(item['keys']), item['word'])) for key, values in index.items()}


def build_normalization_groups(raw_text, reference_text, lexicon_index, bridge_words=('and',)):
    """Group consecutive lexicon words; retain only groups changed by normalization.

    This follows the phoneme scorer's reference-only SequenceMatcher selection and
    whitespace/hyphen/single-connector grouping. A/I cannot match an unchanged
    article/pronoun. Punctuation breaks groups. These are approximate lexicon groups,
    not a semantic entity parser; open-vocabulary URL/name fragments may be uncovered.
    """
    reference_text = unicodedata.normalize('NFKC', reference_text)
    words = tokenize_words(reference_text)
    source = [word['key'] for word in tokenize_words(raw_text)]
    target = [word['key'] for word in words]
    matcher = difflib.SequenceMatcher(a=source, b=target, autojunk=False)
    changed = {i for op, _, _, start, end in matcher.get_opcodes() if op != 'equal' for i in range(start, end)}
    selected = {}
    position = 0
    while position < len(words):
        candidates = []
        for entry in lexicon_index.get(words[position]['key'], []):
            end = position + len(entry['keys'])
            segment = words[position:end]
            if [word['key'] for word in segment] != entry['keys']:
                continue
            if any(not GAP.fullmatch(reference_text[a['end'] : b['start']]) for a, b in zip(segment, segment[1:])):
                continue
            if entry['word'] in {'A', 'I'} and position not in changed:
                continue
            candidates.append(entry)
        if not candidates:
            position += 1
            continue
        length = len(candidates[0]['keys'])
        categories = sorted({c for entry in candidates if len(entry['keys']) == length for c in entry['categories']})
        for i in range(position, position + length):
            selected[i] = categories
        position += length
    candidates, current = [], []
    for i in selected:
        if current:
            between = words[current[-1] + 1 : i]
            adjacent = not between
            bridge = len(between) == 1 and between[0]['key'] in bridge_words
            segment = words[current[-1] : i + 1]
            clean = all(GAP.fullmatch(reference_text[a['end'] : b['start']]) for a, b in zip(segment, segment[1:]))
            if not ((adjacent or bridge) and clean):
                candidates.append(current)
                current = []
        current.append(i)
    if current:
        candidates.append(current)
    groups = []
    for chosen in candidates:
        if not changed.intersection(chosen):
            continue
        first, end = chosen[0], chosen[-1] + 1
        groups.append(
            dict(
                start_word=first,
                end_word=end,
                start_char=words[first]['start'],
                end_char=words[end - 1]['end'],
                reference_text=reference_text[words[first]['start'] : words[end - 1]['end']],
                reference_words=target[first:end],
                categories=sorted({c for i in chosen for c in selected[i]}),
            )
        )
    coverage = dict(
        changed_words=len(changed),
        selected_changed_words=len(changed.intersection(selected)),
        unselected_changed_words=[words[i]['word'] for i in sorted(changed - set(selected))],
    )
    return groups, coverage


def edit_alignment(reference, hypothesis):
    """Minimum Levenshtein word alignment, with deterministic diagonal/delete/insert ties."""
    n, m = len(reference), len(hypothesis)
    directions = bytearray((n + 1) * (m + 1))
    directions[1 : m + 1] = bytes([3]) * m
    for i in range(1, n + 1):
        directions[i * (m + 1)] = 2
    previous = list(range(m + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            diagonal, deletion, insertion = previous[j - 1] + (a != b), previous[j] + 1, current[j - 1] + 1
            best = min(diagonal, deletion, insertion)
            directions[i * (m + 1) + j] = 1 if diagonal == best else 2 if deletion == best else 3
            current.append(best)
        previous = current
    operations, i, j = [], n, m
    while i or j:
        direction = directions[i * (m + 1) + j]
        if direction == 1:
            i, j = i - 1, j - 1
            operations.append(('equal' if reference[i] == hypothesis[j] else 'substitute', i, j))
        elif direction == 2:
            i -= 1
            operations.append(('delete', i, j))
        else:
            j -= 1
            operations.append(('insert', i, j))
    return list(reversed(operations))


def score_text_normalization(raw_text, reference_text, hypothesis_text, lexicon_index):
    """Score full-sentence CER/WER and complete normalization-derived reference groups.

    Empty hypotheses are deletion errors. Insertions inside or immediately beside a
    group are charged once, preferring the right-hand group then the left, as in
    the IPA scorer. Optional 'and', alternative number readings and homophones
    are not silently accepted. CER may exceed 1.0; it is never clipped.
    """
    reference = [word['key'] for word in tokenize_words(reference_text)]
    hypothesis = [word['key'] for word in tokenize_words(hypothesis_text)]
    groups, coverage = build_normalization_groups(raw_text, reference_text, lexicon_index)
    labels = [None] * len(reference)
    for i, group in enumerate(groups):
        for position in range(group['start_word'], group['end_word']):
            labels[position] = i
        group.update(substitutions=0, deletions=0, insertions=0, hypothesis_words=[])
    operations = edit_alignment(reference, hypothesis)
    error_names = {'substitute': 'substitutions', 'delete': 'deletions', 'insert': 'insertions'}
    counts = Counter()
    for kind, i, j in operations:
        if kind != 'equal':
            counts[error_names[kind]] += 1
        if kind == 'insert':
            right = labels[i] if i < len(labels) else None
            left = labels[i - 1] if i else None
            owner = right if right is not None else left
        else:
            owner = labels[i]
        if owner is not None:
            group = groups[owner]
            if kind != 'equal':
                group[error_names[kind]] += 1
            if kind != 'delete':
                group['hypothesis_words'].append(hypothesis[j])
    for group in groups:
        group['hypothesis_text'] = ' '.join(group.pop('hypothesis_words'))
        group['edits'] = group['substitutions'] + group['deletions'] + group['insertions']
        group['exact'] = group['edits'] == 0
        reference_string = ' '.join(group['reference_words'])
        group['reference_characters'] = len(reference_string)
        group['character_edits'] = _distance(reference_string, group['hypothesis_text'])
        group['cer'] = group['character_edits'] / group['reference_characters']
    reference_string, hypothesis_string = ' '.join(reference), ' '.join(hypothesis)
    character_edits = _distance(reference_string, hypothesis_string)
    word_edits = sum(counts.values())
    return dict(
        reference_scored=reference_string,
        hypothesis_scored=hypothesis_string,
        reference_characters=len(reference_string),
        reference_words=len(reference),
        character_edits=character_edits,
        word_edits=word_edits,
        cer=character_edits / len(reference_string) if reference_string else None,
        wer=word_edits / len(reference) if reference else None,
        whole_turn_exact=reference == hypothesis,
        empty_hypothesis=not hypothesis,
        normalization_groups=groups,
        group_count=len(groups),
        group_correct=sum(group['exact'] for group in groups),
        group_exact=sum(group['exact'] for group in groups) / len(groups) if groups else None,
        group_mean_cer=statistics.fmean(group['cer'] for group in groups) if groups else None,
        reference_group_coverage=coverage,
    )


def aggregate_text_scores(scores):
    """Pool full-sentence errors and average group scores equally over eligible turns."""
    scores = list(scores)
    eligible = [score for score in scores if score['group_count']]
    groups = [group for score in eligible for group in score['normalization_groups']]
    characters = sum(score['reference_characters'] for score in scores)
    words = sum(score['reference_words'] for score in scores)
    group_characters = sum(group['reference_characters'] for group in groups)
    return dict(
        turns=len(scores),
        eligible_turns=len(eligible),
        turns_without_groups=len(scores) - len(eligible),
        groups=len(groups),
        correct_groups=sum(group['exact'] for group in groups),
        cer=sum(score['character_edits'] for score in scores) / characters if characters else None,
        wer=sum(score['word_edits'] for score in scores) / words if words else None,
        mean_turn_cer=_mean(score['cer'] for score in scores),
        mean_turn_wer=_mean(score['wer'] for score in scores),
        whole_turn_exact=_mean(score['whole_turn_exact'] for score in scores),
        normalization_group_exact=_mean(score['group_exact'] for score in eligible),
        normalization_group_cer=_mean(score['group_mean_cer'] for score in eligible),
        pooled_group_exact=_mean(group['exact'] for group in groups),
        pooled_group_cer=(
            sum(group['character_edits'] for group in groups) / group_characters if group_characters else None
        ),
        all_groups_correct=_mean(score['group_correct'] == score['group_count'] for score in eligible),
        empty_hypotheses=sum(score['empty_hypothesis'] for score in scores),
        empty_references=sum(score['reference_words'] == 0 for score in scores),
        changed_reference_words=sum(score['reference_group_coverage']['changed_words'] for score in scores),
        selected_changed_reference_words=sum(
            score['reference_group_coverage']['selected_changed_words'] for score in scores
        ),
    )


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _distance(reference, hypothesis):
    try:
        import editdistance
    except ImportError:
        # Standalone use without NeMo dependencies; identical alignment tie policy.
        return sum(kind != 'equal' for kind, _, _ in edit_alignment(reference, hypothesis))
    return editdistance.eval(reference, hypothesis)
