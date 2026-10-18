# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer-independent, exact word/phrase dependencies for text-to-IPA data."""

import hashlib


def resolve_word_alignment(alignment, text, normalized_text, ipa):
    """Return raw/normalized dependencies, or None for unusable alignment.

    Each dependency is (required source-character end, IPA start, IPA end).
    Shared engine spans and complete normalization expressions remain atomic;
    ordinary words stay independent. No boundaries or pronunciations are guessed.
    Offsets and hashes always refer to the original, unmodified strings.
    """
    try:
        if not isinstance(alignment, dict) or alignment.get('dynamic_delay_usable') is False:
            return None
        if alignment.get('offset_unit') != 'unicode_code_points':
            return None
        if alignment.get('ranges') != 'zero_based_end_exclusive':
            return None
        strings = {'text': text, 'text_normalized': normalized_text, 'ipa': ipa}
        if not all(isinstance(value, str) for value in strings.values()) or not ipa.strip():
            return None
        hashes = alignment.get('string_sha256', {})
        if not isinstance(hashes, dict) or any(
            hashes.get(k) != hashlib.sha256(v.encode()).hexdigest() for k, v in strings.items()
        ):
            return None
        source, normalized = alignment['source_words'], alignment['normalized_words']
        _validate_words(source, text)
        _validate_words(normalized, normalized_text)
        ns, nn = len(source), len(normalized)
        if not ns or not nn or len(alignment['normalized_to_ipa']) != nn:
            return None
        mappings = list(alignment.get('source_to_normalized', [None] * ns))
        if len(mappings) != ns:
            return None
        parents = list(range(nn + ns))
        contains_normalized = [True] * nn + [False] * ns

        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def join(indices):
            indices = iter(indices)
            first = next(indices, None)
            if first is not None:
                first = find(first)
                for index in indices:
                    other = find(index)
                    contains_normalized[first] |= contains_normalized[other]
                    parents[other] = first

        spans = list(alignment['normalized_to_ipa'])
        shared_groups = alignment.get('normalized_ipa_shared_groups', []) + alignment.get(
            'source_ipa_shared_groups', []
        )
        for shared in shared_groups:
            nr, ir = shared['normalized_words'], shared['ipa_chars']
            _validate_range(nr, nn, nonempty=True)
            _validate_range(ir, len(ipa))
            join(range(*nr))
            for index in range(*nr):
                if spans[index] is not None:
                    _validate_range(spans[index], len(ipa))
                    if (
                        spans[index][0] < spans[index][1]
                        and ir[0] < ir[1]
                        and (spans[index][1] <= ir[0] or ir[1] <= spans[index][0])
                    ):
                        return None
                    spans[index] = [min(ir[0], spans[index][0]), max(ir[1], spans[index][1])]
                else:
                    spans[index] = ir
        for index, span in enumerate(spans):
            if span is None:
                if normalized_text[slice(*normalized[index])].strip():
                    return None
                spans[index] = [0, 0]  # an explicitly whitespace-only placeholder
            _validate_range(spans[index], len(ipa))

        # Overlapping IPA spans explicitly share a pronunciation. Connect the
        # corresponding words even if the exporter omitted a redundant group.
        intervals = sorted((a, b, i) for i, (a, b) in enumerate(spans) if a < b)
        end, previous = -1, None
        for start, stop, index in intervals:
            if start < end:
                join((previous, index))
            if stop > end:
                end, previous = stop, index
        cursor = 0
        for start, stop, _ in intervals:
            if start > cursor and ipa[cursor:start].strip():
                return None
            cursor = max(cursor, stop)
        if ipa[cursor:].strip():
            return None

        def components():
            result = {}
            for index in range(nn):
                result.setdefault(find(index), []).append(index)
            return result

        def close_ranges():
            # A phrase may connect across another alignment group. Include all
            # intervening words rather than inventing an internal boundary.
            while True:
                changed = False
                for offset, count in ((0, nn), (nn, ns)):
                    bounds = {}
                    for i in range(count):
                        key = find(offset + i)
                        if key not in bounds:
                            bounds[key] = [i, i]
                        else:
                            bounds[key][1] = i
                    for first, last in bounds.values():
                        for i in range(first, last + 1):
                            if find(offset + i) != find(offset + first):
                                join((offset + first, offset + i))
                                changed = True
                if not changed:
                    return

        close_ranges()
        normalized_groups = []
        for indices in components().values():
            pieces = [spans[i] for i in indices if ipa[slice(*spans[i])].strip()]
            if pieces:
                normalized_groups.append(
                    (normalized[indices[-1]][1], min(p[0] for p in pieces), max(p[1] for p in pieces))
                )

        groups = alignment.get('groups', [])
        regions = alignment.get('unresolved_regions', [])
        for group in groups:
            sr, nr = group['source_words'], group['normalized_words']
            _validate_range(sr, ns)
            if nr is None:
                region = next(
                    (
                        region
                        for region in regions
                        if region['source_words'][0] <= sr[0] <= sr[1] <= region['source_words'][1]
                    ),
                    None,
                )
                if region is None:
                    return None
                nr = region['normalized_words']
            _validate_range(nr, nn)
            for key, length in (
                ('source_chars', len(text)),
                ('normalized_chars', len(normalized_text)),
                ('ipa_chars', len(ipa)),
            ):
                if group.get(key) is not None:
                    _validate_range(group[key], length)
            if (
                group.get('mapping') == 'one_to_one'
                and group.get('text_status', group.get('status')) == 'ok'
                and sr[1] - sr[0] == nr[1] - nr[0]
            ):
                for si, ni in zip(range(*sr), range(*nr)):
                    if mappings[si] is None:
                        mappings[si] = [ni, ni + 1]
            if group.get('changed') or group.get('entity_type'):
                join(list(range(*nr)) + [nn + i for i in range(*sr)])
        for group in regions:
            sr, nr = group['source_words'], group['normalized_words']
            _validate_range(sr, ns, nonempty=True)
            _validate_range(nr, nn, nonempty=True)
            join(list(range(*nr)) + [nn + i for i in range(*sr)])
            for i in range(*sr):
                if mappings[i] is None:
                    mappings[i] = nr
        for group in alignment.get('source_ipa_shared_groups', []):
            sr, nr = group['source_words'], group['normalized_words']
            _validate_range(sr, ns, nonempty=True)
            join(list(range(*nr)) + [nn + i for i in range(*sr)])
        for index, mapping in enumerate(mappings):
            if mapping is None:
                if contains_normalized[find(nn + index)]:
                    continue  # an explicit complete phrase already owns this word
                return None
            _validate_range(mapping, nn)
            join([nn + index] + list(range(*mapping)))
        close_ranges()
        source_ends = {}
        for index, word in enumerate(source):
            key = find(nn + index)
            source_ends[key] = max(word[1], source_ends.get(key, 0))
        raw_groups = []
        for key, indices in components().items():
            pieces = [spans[i] for i in indices if ipa[slice(*spans[i])].strip()]
            if pieces:
                if key not in source_ends:
                    return None
                raw_groups.append((source_ends[key], min(p[0] for p in pieces), max(p[1] for p in pieces)))
        return {
            'source': tuple(sorted(raw_groups, key=lambda g: g[1])),
            'normalized': tuple(sorted(normalized_groups, key=lambda g: g[1])),
        }
    except (KeyError, TypeError, IndexError, ValueError):
        return None


def _validate_range(span, length, nonempty=False):
    if not (
        isinstance(span, (list, tuple))
        and len(span) == 2
        and all(type(x) is int for x in span)
        and 0 <= span[0] <= span[1] <= length
        and (not nonempty or span[0] < span[1])
    ):
        raise ValueError('Invalid alignment range')


def _validate_words(words, text):
    previous = 0
    for word in words:
        _validate_range(word, len(text))
        if word[0] < previous:
            raise ValueError('Overlapping or unordered words')
        previous = word[1]
