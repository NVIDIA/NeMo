# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU scheduling of supervised phoneme PAD/wait targets from word alignment."""

import hashlib
import re
from bisect import bisect_left, bisect_right


def phoneme_source_dependencies(alignment, text, normalized_text, ipa, selected_text, text_offsets, ipa_offsets):
    """Map IPA-token indices to the last required visible text-token index.

    Offsets refer to each complete string, before manually added BOS/EOS. A text
    index equal to len(text_offsets) means the manually appended text EOS.
    Missing, stale or unresolvable metadata conservatively waits for text EOS
    before the first IPA token. Returns (dependencies, full_text_fallback).
    """
    fallback = ({0: len(text_offsets)}, True)
    if not isinstance(alignment, dict) or not ipa_offsets:
        return fallback
    if alignment.get('offset_unit') != 'unicode_code_points':
        return fallback
    if alignment.get('ranges') != 'zero_based_end_exclusive':
        return fallback
    strings = {'text': text, 'text_normalized': normalized_text, 'ipa': ipa}
    hashes = alignment.get('string_sha256', {})
    if not isinstance(hashes, dict) or not all(isinstance(v, str) for v in strings.values()):
        return fallback
    if any(hashes.get(k) != hashlib.sha256(v.encode('utf-8')).hexdigest() for k, v in strings.items()):
        return fallback

    if selected_text == text:
        source_key, original = 'source_chars', text
    elif selected_text == normalized_text:
        source_key, original = 'normalized_chars', normalized_text
    elif selected_text == _strip_with_origins(text)[0]:
        source_key, original = 'source_chars', text
    else:
        return fallback
    origins = None if selected_text == original else _strip_with_origins(original)[1]

    try:
        groups = alignment['groups']
        source_words, normalized_words = alignment['source_words'], alignment['normalized_words']
        candidates = []
        covered_regions = []
        # Recover a coarse but exact many-to-many span, e.g. "5 years full-time".
        for region in alignment.get('unresolved_regions', []):
            sr, nr = region['source_words'], region['normalized_words']
            _check_range(sr, len(source_words))
            _check_range(nr, len(normalized_words))
            pieces = []
            for i in range(*nr):
                span = alignment['normalized_to_ipa'][i]
                if span is None:
                    shared = next(
                        (
                            g
                            for g in alignment.get('normalized_ipa_shared_groups', [])
                            if g['normalized_words'][0] <= i < g['normalized_words'][1]
                        ),
                        None,
                    )
                    if (
                        shared is None
                        or not nr[0] <= shared['normalized_words'][0] <= shared['normalized_words'][1] <= nr[1]
                    ):
                        return fallback
                    span = shared['ipa_chars']
                _check_range(span, len(ipa))
                pieces.append(span)
            if not pieces:
                return fallback
            candidates.append(
                {
                    'source_chars': [source_words[sr[0]][0], source_words[sr[1] - 1][1]],
                    'normalized_chars': [normalized_words[nr[0]][0], normalized_words[nr[1] - 1][1]],
                    'ipa_chars': [min(p[0] for p in pieces), max(p[1] for p in pieces)],
                }
            )
            covered_regions.append(sr)

        for group in groups:
            if group.get('status') != 'ok':
                sr = group['source_words']
                if not any(r[0] <= sr[0] <= sr[1] <= r[1] for r in covered_regions):
                    return fallback
                continue
            if not (group.get('changed') or group.get('entity_type')):
                continue
            ir = group.get('ipa_chars')
            if ir is None:
                return fallback
            _check_range(ir, len(ipa))
            if ir[0] < ir[1]:
                candidates.append(group)

        if source_key == 'normalized_chars':
            # Normalized input is already expanded: each spoken word only
            # depends on that word, rather than the entire original number.
            word_candidates = []
            for group in candidates:
                for i, word in enumerate(normalized_words):
                    if word[0] >= group['normalized_chars'][1] or word[1] <= group['normalized_chars'][0]:
                        continue
                    ir = alignment['normalized_to_ipa'][i]
                    if ir is None:
                        shared = next(
                            (
                                g
                                for g in alignment.get('normalized_ipa_shared_groups', [])
                                if g['normalized_words'][0] <= i < g['normalized_words'][1]
                            ),
                            None,
                        )
                        if shared is None:
                            return fallback
                        ir = shared['ipa_chars']
                        word = [word[0], normalized_words[shared['normalized_words'][1] - 1][1]]
                    if ir[0] < ir[1]:
                        word_candidates.append({'normalized_chars': word, 'ipa_chars': ir})
            candidates = word_candidates

        dependencies = {}
        ipa_ends = [end for _, end in ipa_offsets]
        text_ends = [end for _, end in text_offsets]
        if ipa_ends != sorted(ipa_ends) or text_ends != sorted(text_ends):
            return fallback
        for group in candidates:
            span = list(group['ipa_chars'])
            source_end = group[source_key][1]
            # A shared engine boundary can cross the requested expression. Wait
            # for the entire shared group before emitting its first IPA token.
            for shared in alignment.get('source_ipa_shared_groups', []):
                ir = shared['ipa_chars']
                if ir[0] < span[1] and span[0] < ir[1]:
                    span = [min(span[0], ir[0]), max(span[1], ir[1])]
                    key, words = (
                        ('source_words', source_words)
                        if source_key == 'source_chars'
                        else ('normalized_words', normalized_words)
                    )
                    source_end = max(source_end, words[shared[key][1] - 1][1])
            _check_range([0, source_end], len(original))
            end = source_end if origins is None else bisect_right(origins, source_end)
            if end == len(selected_text):
                ready = len(text_offsets)  # EOS proves the final word is complete.
            else:
                # Include the next character so the word/expression boundary is
                # actually visible, including a space carried by the next BPE.
                ready = bisect_left(text_ends, end + 1)
                if ready < len(text_ends):
                    ready = bisect_right(text_ends, text_ends[ready]) - 1
            first = bisect_right(ipa_ends, span[0])
            if first >= len(ipa_offsets) or ipa_offsets[first][0] >= span[1]:
                return fallback
            # If a BPE crosses the expression boundary, gate that entire BPE.
            dependencies[first] = max(ready, dependencies.get(first, -1))
        return dependencies, False
    except (KeyError, TypeError, IndexError, ValueError):
        return fallback


def insert_phoneme_waits(token_ids, dependencies, pad_id, loss_mask=None):
    """Insert waits before targets whose source is not yet causally visible.

    token_ids includes phoneme BOS/EOS. Dependency keys index these targets;
    values index the complete text channel, including its optional BOS/EOS.
    Target j is predicted at j - 1 + all preceding waits. Dynamic mode has no
    fixed phoneme-channel delay.
    Returns (new IDs, remapped optional loss mask, number of inserted waits).
    """
    if pad_id is None:
        raise ValueError('Dynamic phoneme delay requires a phoneme PAD ID')
    if loss_mask is not None and len(loss_mask) != len(token_ids):
        raise ValueError('Phoneme loss mask must match token IDs')
    output, mask, waits = [], [] if loss_mask is not None else None, 0
    for index, token in enumerate(token_ids):
        extra = max(0, dependencies.get(index, -1) - (index - 1 + waits)) if index else 0
        output.extend([pad_id] * extra)
        output.append(token)
        if mask is not None:
            mask.extend([False] * extra)
            mask.append(loss_mask[index])
        waits += extra
    return output, mask, waits


def _strip_with_origins(text):
    """Match the dataset's timestamp removal/whitespace collapse with end offsets."""
    chars, origins, cursor = [], [], 0
    for match in re.finditer(r'<\|\d+\|>', text):
        for i in range(cursor, match.start()):
            chars.append(text[i])
            origins.append(i + 1)
        cursor = match.end()
    for i in range(cursor, len(text)):
        chars.append(text[i])
        origins.append(i + 1)
    result, result_origins = [], []
    for char, origin in zip(chars, origins):
        if char.isspace():
            if result and result[-1] == ' ':
                result_origins[-1] = origin
            elif result:
                result.append(' ')
                result_origins.append(origin)
        else:
            result.append(char)
            result_origins.append(origin)
    if result and result[-1] == ' ':
        result.pop()
        result_origins.pop()
    return ''.join(result), result_origins


def _check_range(span, length):
    if not (
        isinstance(span, (list, tuple))
        and len(span) == 2
        and all(type(x) is int for x in span)
        and 0 <= span[0] <= span[1] <= length
    ):
        raise ValueError('Invalid alignment range')
