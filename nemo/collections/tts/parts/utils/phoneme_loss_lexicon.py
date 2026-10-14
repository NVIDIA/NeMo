# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU matching of normalization vocabulary to IPA BPE loss positions."""

import re
import unicodedata
from pathlib import Path

import torch
import yaml


class PhonemeLossLexicon:
    """Compile word/phrase pronunciations once per data-loader worker.

    Matches use complete IPA words, ignoring primary/secondary stress. Matching
    the full IPA string before projecting character spans to BPE offsets avoids
    dependence on standalone-word BPE segmentation. No predicted tokens or
    input-to-output alignment are used. IPA homophones remain indistinguishable.
    """

    def __init__(self, lexicon_path):
        with open(lexicon_path, encoding='utf-8') as input_file:
            data = yaml.safe_load(input_file)
        if not isinstance(data, dict) or data.get('schema_version') != 1:
            raise ValueError('Phoneme loss lexicon requires schema_version: 1')
        languages = data.get('languages')
        if not isinstance(languages, list) or not languages or not all(isinstance(x, str) and x for x in languages):
            raise ValueError('Phoneme loss lexicon requires a nonempty languages list')
        entries = data.get('entries')
        if not isinstance(entries, list) or not entries:
            raise ValueError('Phoneme loss lexicon requires nonempty entries')
        self.path = str(Path(lexicon_path))
        self.languages = frozenset(_language_family(x) for x in languages)
        self._trie = {}
        self._pronunciations = []
        self._bound_tokenizer = None
        self.tokenized_pronunciations = {}
        seen_words = set()
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get('word'), str) or not entry['word'].strip():
                raise ValueError('Every phoneme loss entry requires a nonempty word')
            word = entry['word']
            if word in seen_words:
                raise ValueError(f'Duplicate phoneme loss lexicon word: {word}')
            seen_words.add(word)
            pronunciations = entry.get('pronunciations')
            if not isinstance(pronunciations, dict) or not pronunciations:
                raise ValueError(f'Missing pronunciation mapping for {word!r}')
            for voice, variants in pronunciations.items():
                if not isinstance(variants, list) or not variants:
                    raise ValueError(f'Pronunciations for {word!r}/{voice} must be a nonempty list')
                for ipa in variants:
                    if not isinstance(ipa, str) or not ipa.strip():
                        raise ValueError(f'Empty pronunciation for {word!r}/{voice}')
                    pieces = [_canonical_ipa(match.group()) for match in _IPA_WORD.finditer(ipa)]
                    if not pieces or not all(pieces) or len(pieces) > 64:
                        raise ValueError(f'Invalid pronunciation length for {word!r}/{voice}')
                    node = self._trie
                    for piece in pieces:
                        node = node.setdefault(piece, {})
                    node[None] = True
                    self._pronunciations.append((word, voice, ipa))

    def supports_language(self, language):
        return _language_family(language) in self.languages

    def bind_tokenizer(self, tokenizer):
        """Tokenize lexicon variants once, validating the selected IPA tokenizer."""
        if tokenizer is self._bound_tokenizer:
            return
        if not callable(getattr(tokenizer, 'encode_with_offsets', None)):
            raise ValueError('Phoneme lexicon loss weighting requires an IPA BPE tokenizer with character offsets')
        compiled = {}
        for word, voice, ipa in self._pronunciations:
            ids, _ = tokenizer.encode_with_offsets(ipa)
            if not ids:
                raise ValueError(f'IPA tokenizer produced no IDs for {word!r}/{voice}')
            compiled.setdefault(word, {}).setdefault(voice, []).append(tuple(ids))
        self.tokenized_pronunciations = compiled
        self._bound_tokenizer = tokenizer

    def find_spans(self, ipa):
        """Return the union of matching character spans in the original IPA."""
        words = list(_IPA_WORD.finditer(ipa))
        canonical = [_canonical_ipa(word.group()) for word in words]
        spans = []
        for start in range(len(words)):
            node = self._trie
            for end in range(start, len(words)):
                if end > start and not ipa[words[end - 1].end() : words[end].start()].isspace():
                    break  # A phrase must not cross punctuation or turn boundaries.
                node = node.get(canonical[end])
                if node is None:
                    break
                if None in node:
                    spans.append((words[start].start(), words[end].end()))
        merged = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        return merged

    def encode_with_mask(self, ipa, tokenizer):
        """Encode once; mark BPE tokens overlapping a matched IPA word/phrase."""
        self.bind_tokenizer(tokenizer)
        ids, offsets = tokenizer.encode_with_offsets(ipa)
        spans = self.find_spans(ipa)
        mask = [False] * len(ids)
        span_index = 0
        for token_index, (start, end) in enumerate(offsets):
            while span_index < len(spans) and spans[span_index][1] <= start:
                span_index += 1
            if span_index < len(spans) and end > start:
                mask[token_index] = start < spans[span_index][1] and end > spans[span_index][0]
        return ids, mask


def stack_phoneme_loss_mask(mask, phoneme_tokens, stacking_factor, bos_id):
    """Follow ``stack_codes`` and its target shift, without weighting BOS/EOS.

    Both inputs have shape (B, L). The result has shape (B, S, T_target).
    The initial BOS is removed before stacking when ``stack_codes`` does so.
    Padding and partial final stacks have a false mask.
    """
    if mask is None or mask.shape != phoneme_tokens.shape or mask.dtype != torch.bool:
        raise ValueError('phoneme_loss_mask must be Boolean and match the unstacked phoneme token shape')
    if stacking_factor == 1:
        return mask.unsqueeze(1)[:, :, 1:]
    contains_bos = phoneme_tokens[0, 0].item() == bos_id
    if contains_bos:
        mask = mask[:, 1:]
    mask = torch.nn.functional.pad(mask, (0, (-mask.size(1)) % stacking_factor), value=False)
    stacked = mask.reshape(mask.size(0), -1, stacking_factor).transpose(1, 2)
    if contains_bos:
        stacked = torch.cat([stacked.new_zeros(stacked.size(0), stacking_factor, 1), stacked], dim=2)
    return stacked[:, :, 1:]


def make_phoneme_loss_weights(mask, phoneme_tokens, stacking_factor, bos_id, scale=1.0, mode='train'):
    """Return training-only per-target weights; the disabled path performs no tensor work."""
    if mode != 'train' or scale == 1.0:
        return None
    if mask is None:
        raise ValueError('Enabled phoneme lexicon loss requires phoneme_loss_mask from the dataset')
    stacked_mask = stack_phoneme_loss_mask(mask, phoneme_tokens, stacking_factor, bos_id)
    return 1.0 + (scale - 1.0) * stacked_mask.float()


# eSpeak-NG IPA separates words by whitespace. Punctuation separates matches;
# IPA stress and length symbols remain part of the word until canonicalization.
_IPA_WORD = re.compile(r'[^\s.,;:!?()\[\]{}<>/\\|_=#@+%*\-\"\']+')
_STRESS = str.maketrans('', '', 'ˈˌ')


def _canonical_ipa(ipa):
    return unicodedata.normalize('NFC', ipa.translate(_STRESS))


def _language_family(language):
    return str(language or '').lower().replace('_', '-').split('-')[0]
