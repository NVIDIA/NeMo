# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU scheduling of supervised phoneme PAD/wait targets from word alignment."""

import hashlib
import re
from bisect import bisect_left, bisect_right


def phoneme_source_dependencies(
    alignment,
    text,
    normalized_text,
    ipa,
    selected_text,
    text_offsets,
    ipa_offsets,
    *,
    selected_ipa=None,
    resolved_alignment=None,
):
    """Map every spoken word/phrase to its required visible source BPE.

    Offsets and metadata hashes refer to complete original strings. Returns
    (dependencies, unusable); unusable records must be excluded before batching,
    never converted into whole-text-EOS waiting targets. Dependencies use no
    fixed delay, and insert_phoneme_waits counts all earlier emitted targets.
    """
    from nemo.collections.common.data.lhotse.text_alignment import resolve_word_alignment

    unusable = ({}, True)
    strings = {'text': text, 'text_normalized': normalized_text, 'ipa': ipa}
    if not isinstance(alignment, dict) or not all(isinstance(v, str) for v in strings.values()):
        return unusable
    hashes = alignment.get('string_sha256', {})
    if not isinstance(hashes, dict) or any(
        hashes.get(key) != hashlib.sha256(value.encode('utf-8')).hexdigest() for key, value in strings.items()
    ):
        return unusable
    if resolved_alignment is None:
        resolved_alignment = resolve_word_alignment(alignment, text, normalized_text, ipa)
    if resolved_alignment is None or not ipa_offsets:
        return unusable
    if selected_text in (text, _strip_with_origins(text)[0]):
        source_key, original = 'source', text
    elif selected_text in (normalized_text, _strip_with_origins(normalized_text, strip_timestamps=False)[0]):
        source_key, original = 'normalized', normalized_text
    else:
        return unusable
    origins = (
        None
        if selected_text == original
        else _strip_with_origins(original, strip_timestamps=source_key == 'source')[1]
    )
    ipa_origins = None
    if selected_ipa is not None and selected_ipa != ipa:
        compact_ipa, ipa_origins = _strip_with_origins(ipa, strip_timestamps=False)
        if selected_ipa != compact_ipa:
            return unusable
    try:
        dependencies = {}
        ipa_ends = [end for _, end in ipa_offsets]
        text_ends = [end for _, end in text_offsets]
        if ipa_ends != sorted(ipa_ends) or text_ends != sorted(text_ends):
            return unusable
        for source_end, ipa_start, ipa_end in resolved_alignment[source_key]:
            _check_range([ipa_start, ipa_end], len(ipa))
            while ipa_start < ipa_end and ipa[ipa_start].isspace():
                ipa_start += 1
            while ipa_start < ipa_end and ipa[ipa_end - 1].isspace():
                ipa_end -= 1
            if ipa_start == ipa_end:
                continue
            if ipa_origins is not None:
                ipa_start, ipa_end = (bisect_right(ipa_origins, pos) for pos in (ipa_start, ipa_end))
            _check_range([0, source_end], len(original))
            while source_end and original[source_end - 1].isspace():
                source_end -= 1
            end = source_end if origins is None else bisect_right(origins, source_end)
            if end == len(selected_text):
                ready = len(text_offsets)  # EOS proves the final word is complete.
            else:
                # A separator may be carried by the following BPE. Consume all
                # byte-fallback pieces ending at that character boundary.
                ready = bisect_left(text_ends, end + 1)
                if ready < len(text_ends):
                    ready = bisect_right(text_ends, text_ends[ready]) - 1
            first = bisect_right(ipa_ends, ipa_start)
            if first >= len(ipa_offsets) or ipa_offsets[first][0] >= ipa_end:
                return unusable
            # An IPA BPE crossing word boundaries depends on every word it spans.
            dependencies[first] = max(ready, dependencies.get(first, -1))
        return dependencies, False
    except (KeyError, TypeError, IndexError, ValueError):
        return unusable


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


def _strip_with_origins(text, strip_timestamps=True):
    """Match the dataset's timestamp removal/whitespace collapse with end offsets."""
    chars, origins, cursor = [], [], 0
    for match in re.finditer(r'<\|\d+\|>', text) if strip_timestamps else ():
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
