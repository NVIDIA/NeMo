# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import itertools
import re
import string
import unicodedata
from builtins import str as unicode
from typing import List

# Example of parsing by groups via _words_re.
# Groups:
# 1st group -- valid english words,
# 2nd group -- any substring starts from | to | (mustn't be nested), useful when you want to leave sequence unchanged,
# 3rd group -- punctuation marks.
# Text (first line) and mask of groups for every char (second line).
# config file must contain |EY1 EY1|, B, C, D, E, F, and G.
# 111111311113111131111111322222222233133133133133133111313
_words_re = re.compile("([a-zA-Z]+(?:[a-zA-Z-']*[a-zA-Z]+)*)|(\|[^|]*\|)|([^a-zA-Z|]+)")


def english_text_preprocessing(text):
    text = unicode(text)
    text = ''.join(char for char in unicodedata.normalize('NFD', text) if unicodedata.category(char) != 'Mn')
    return text


def english_word_tokenize(text):
    """
    Convert text (str) to List[Tuple[Union[str, List[str]], bool]] where every tuple denotes word representation and flag whether to leave unchanged or not.
    Word can be one of: valid english word, any substring starts from | to | (unchangeable word) or punctuation marks.
    This function expects that unchangeable word is carefully divided by spaces (e.g. HH AH L OW).
    Unchangeable word will be splitted by space and represented as List[str], other cases are represented as str.
    """
    words = _words_re.findall(text)
    result = []
    for word in words:
        maybe_word, maybe_without_changes, maybe_punct = word

        if maybe_word != '':
            without_changes = False
            result.append((maybe_word.lower(), without_changes))
        elif maybe_punct != '':
            without_changes = False
            result.append((re.sub(r'\s(\d)', r'\1', maybe_punct.upper()), without_changes))
        elif maybe_without_changes != '':
            without_changes = True
            result.append((maybe_without_changes[1:-1].split(" "), without_changes))
    return result


class BaseTokenizer(abc.ABC):
    PAD, BLANK, OOV = '<pad>', '<blank>', '<oov>'

    def __init__(self, tokens, *, pad=PAD, blank=BLANK, oov=OOV, sep='', add_blank_at=None):
        """Abstract class for creating an arbitrary tokenizer to convert string to list of int tokens.
        Args:
            tokens: List of tokens.
            pad: Pad token as string.
            blank: Blank token as string.
            oov: OOV token as string.
            sep: Separation token as string.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
        """
        super().__init__()

        tokens = list(tokens)
        self.pad, tokens = len(tokens), tokens + [pad]  # Padding

        if add_blank_at is not None:
            self.blank, tokens = len(tokens), tokens + [blank]  # Reserved for blank from asr-model
        else:
            # use add_blank_at=None only for ASR where blank is added automatically, disable blank here
            self.blank = None

        self.oov, tokens = len(tokens), tokens + [oov]  # Out Of Vocabulary

        if add_blank_at == "last":
            tokens[-1], tokens[-2] = tokens[-2], tokens[-1]
            self.oov, self.blank = self.blank, self.oov

        self.tokens = tokens
        self.sep = sep

        self._util_ids = {self.pad, self.blank, self.oov}
        self._token2id = {l: i for i, l in enumerate(tokens)}
        self._id2token = tokens

    def __call__(self, text: str) -> List[int]:
        return self.encode(text)

    @abc.abstractmethod
    def encode(self, text: str) -> List[int]:
        """Turns str text into int tokens."""
        pass

    def decode(self, tokens: List[int]) -> str:
        """Turns ints tokens into str text."""
        return self.sep.join(self._id2token[t] for t in tokens if t not in self._util_ids)


class EnglishCharsTokenizer(BaseTokenizer):
    # fmt: off
    PUNCT_LIST = (  # Derived from LJSpeech and "/" additionally
        ',', '.', '!', '?', '-',
        ':', ';', '/', '"', '(',
        ')', '[', ']', '{', '}',
    )
    # fmt: on

    def __init__(
        self,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=None,
        text_preprocessing_func=english_text_preprocessing,
        word_tokenize_func=english_word_tokenize,
    ):
        """English char-based tokenizer.
        Args:
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            text_preprocessing_func: Function for preprocessing raw text.
            word_tokenize_func: Function for tokenizing text to words.
        """

        tokens = []
        self.space, tokens = len(tokens), tokens + [' ']  # Space
        tokens.extend(string.ascii_lowercase)
        if apostrophe:
            tokens.append("'")  # Apostrophe for saving "don't" and "Joe's"

        if punct:
            if non_default_punct_list is not None:
                self.PUNCT_LIST = non_default_punct_list
            tokens.extend(self.PUNCT_LIST)

        super().__init__(tokens, add_blank_at=add_blank_at)

        self.punct = punct
        self.pad_with_space = pad_with_space

        self.text_preprocessing_func = text_preprocessing_func
        self.word_tokenize_func = word_tokenize_func

    def encode(self, text):
        """See base class."""
        cs, space, tokens = [], self.tokens[self.space], set(self.tokens)

        words = [
            word[0] if isinstance(word, tuple) else word
            for word in self.word_tokenize_func(self.text_preprocessing_func(text))
        ]
        for c in "".join(words):  # noqa
            # Add space if last one isn't one
            if c == space and len(cs) > 0 and cs[-1] != space:
                cs.append(c)
            # Add next char
            elif (c.isalnum() or c == "'") and c in tokens:
                cs.append(c)
            # Add punct
            elif (c in self.PUNCT_LIST) and self.punct:
                cs.append(c)

        # Remove trailing spaces
        while cs[-1] == space:
            cs.pop()

        if self.pad_with_space:
            cs = [space] + cs + [space]

        return [self._token2id[p] for p in cs]


class EnglishPhonemesTokenizer(BaseTokenizer):
    # fmt: off
    PUNCT_LIST = (  # Derived from LJSpeech and "/" additionally
        ',', '.', '!', '?', '-',
        ':', ';', '/', '"', '(',
        ')', '[', ']', '{', '}',
    )
    VOWELS = (
        'AA', 'AE', 'AH', 'AO', 'AW',
        'AY', 'EH', 'ER', 'EY', 'IH',
        'IY', 'OW', 'OY', 'UH', 'UW',
    )
    CONSONANTS = (
        'B', 'CH', 'D', 'DH', 'F', 'G',
        'HH', 'JH', 'K', 'L', 'M', 'N',
        'NG', 'P', 'R', 'S', 'SH', 'T',
        'TH', 'V', 'W', 'Y', 'Z', 'ZH',
    )
    # fmt: on

    def __init__(
        self,
        g2p,
        punct=True,
        non_default_punct_list=None,
        stresses=False,
        chars=False,
        *,
        space=' ',
        silence=None,
        apostrophe=True,
        oov=BaseTokenizer.OOV,
        sep='|',  # To be able to distinguish between 2/3 letters codes.
        add_blank_at=None,
        pad_with_space=False,
    ):
        """English phoneme-based tokenizer.
        Args:
            g2p: Grapheme to phoneme module.
            punct: Whether to reserve grapheme for basic punctuation or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            stresses: Whether to use phonemes codes with stresses (0-2) or not.
            chars: Whether to additionally use chars together with phonemes. It is useful if g2p module can return chars too.
            space: Space token as string.
            silence: Silence token as string (will be disabled if it is None).
            apostrophe: Whether to use apostrophe or not.
            oov: OOV token as string.
            sep: Separation token as string.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
        """

        tokens = []
        self.space, tokens = len(tokens), tokens + [space]  # Space

        if silence is not None:
            self.silence, tokens = len(tokens), tokens + [silence]  # Silence

        tokens.extend(self.CONSONANTS)
        vowels = list(self.VOWELS)

        if stresses:
            vowels = [f'{p}{s}' for p, s in itertools.product(vowels, (0, 1, 2))]
        tokens.extend(vowels)

        if chars:
            tokens.extend(string.ascii_lowercase)

        if apostrophe:
            tokens.append("'")  # Apostrophe

        if punct:
            if non_default_punct_list is not None:
                self.PUNCT_LIST = non_default_punct_list
            tokens.extend(self.PUNCT_LIST)

        super().__init__(tokens, oov=oov, sep=sep, add_blank_at=add_blank_at)

        self.chars = chars
        self.punct = punct
        self.stresses = stresses
        self.pad_with_space = pad_with_space

        self.g2p = g2p

    def encode(self, text):
        """See base class."""
        ps, space, tokens = [], self.tokens[self.space], set(self.tokens)

        for p in self.g2p(text):  # noqa
            # Remove stress
            if p.isalnum() and len(p) == 3 and not self.stresses:
                p = p[:2]

            # Add space if last one isn't one
            if p == space and len(ps) > 0 and ps[-1] != space:
                ps.append(p)
            # Add next phoneme or char (if chars=True)
            elif (p.isalnum() or p == "'") and p in tokens:
                ps.append(p)
            # Add punct
            elif (p in self.PUNCT_LIST) and self.punct:
                ps.append(p)

        # Remove trailing spaces
        while ps[-1] == space:
            ps.pop()

        if self.pad_with_space:
            ps = [space] + ps + [space]

        return [self._token2id[p] for p in ps]
