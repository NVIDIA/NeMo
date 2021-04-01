# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import string
from typing import List

import nltk

from nemo.collections.asr.parts import parsers

try:
    nltk.data.find('taggers/averaged_perceptron_tagger.zip')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('corpora/cmudict.zip')
except LookupError:
    nltk.download('cmudict', quiet=True)

try:
    import g2p_en  # noqa

    _g2p = g2p_en.G2p()
    _g2p.variables = None

    HAVE_G2P = True
except (FileNotFoundError, LookupError):
    HAVE_G2P = False


class Base(abc.ABC):
    """Vocabulary for turning str text to list of int tokens."""

    # fmt: off
    PUNCT = (  # Derived from LJSpeech
        ',', '.', '!', '?', '-',
        ':', ';', '/', '"', '(',
        ')', '[', ']', '{', '}',
    )
    # fmt: on
    PAD, BLANK, OOV = '<pad>', '<blank>', '<oov>'

    def __init__(self, labels, *, pad=PAD, blank=BLANK, oov=OOV, sep='', add_blank_at="last_but_one"):
        super().__init__()

        labels = list(labels)
        self.pad, labels = len(labels), labels + [pad]  # Padding

        if add_blank_at is not None:
            self.blank, labels = len(labels), labels + [blank]  # Reserved for blank from QN
        else:
            # use add_blank_at=None only for ASR where blank is added automatically
            self.blank = -1

        self.oov, labels = len(labels), labels + [oov]  # Out Of Vocabulary

        if add_blank_at == "last":
            labels[-1], labels[-2] = labels[-2], labels[-1]
            self.oov, self.blank = self.blank, self.oov

        self.labels = labels
        self.sep = sep

        self._util_ids = {self.pad, self.blank, self.oov}
        self._label2id = {l: i for i, l in enumerate(labels)}
        self._id2label = labels

    def __call__(self, text: str) -> List[int]:
        return self.encode(text)

    @abc.abstractmethod
    def encode(self, text: str) -> List[int]:
        """Turns str text into int tokens."""
        pass

    def decode(self, tokens: List[int]) -> str:
        """Turns ints tokens into str text."""
        return self.sep.join(self._id2label[t] for t in tokens if t not in self._util_ids)


class Chars(Base):
    """Chars vocabulary."""

    def __init__(
        self, punct=True, spaces=False, apostrophe=True, add_blank_at="last_but_one",
    ):
        labels = []
        self.space, labels = len(labels), labels + [' ']  # Space
        labels.extend(string.ascii_lowercase)
        if apostrophe:
            labels.append("'")  # Apostrophe for saving "don't" and "Joe's"

        if punct:
            labels.extend(self.PUNCT)

        super().__init__(labels, add_blank_at=add_blank_at)

        self.punct = punct
        self.spaces = spaces

        self._parser = parsers.ENCharParser(labels)

    def encode(self, text):
        """See base class."""
        text = self._parser._normalize(text)  # noqa

        if self.spaces:
            for p in set(text) & set(self.PUNCT):
                text = text.replace(p, f' {p} ')
            text = text.strip().replace('  ', ' ')

        return self._parser._tokenize(text)  # noqa


class Phonemes(Base):
    """Phonemes vocabulary."""

    # fmt: off
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
        punct=True,
        stresses=False,
        spaces=True,
        *,
        space=' ',
        silence=None,
        apostrophe=True,
        oov=Base.OOV,
        sep='|',  # To be able to distinguish between 2/3 letters codes.
        add_blank_at="last_but_one",
    ):
        labels = []
        self.space, labels = len(labels), labels + [space]  # Space
        if silence is not None:
            self.silence, labels = len(labels), labels + [silence]  # Silence
        labels.extend(self.CONSONANTS)
        vowels = list(self.VOWELS)
        if stresses:
            vowels = [f'{p}{s}' for p, s in itertools.product(vowels, (0, 1, 2))]
        labels.extend(vowels)
        if apostrophe:
            labels.append("'")  # Apostrophe

        if punct:
            labels.extend(self.PUNCT)

        super().__init__(labels, oov=oov, sep=sep, add_blank_at=add_blank_at)

        self.punct = punct
        self.stresses = stresses
        self.spaces = spaces

    def encode(self, text):
        """See base class."""
        ps, space, labels = [], self.labels[self.space], set(self.labels)

        for p in _g2p(text):  # noqa
            # Remove stress
            if p.isalnum() and len(p) == 3 and not self.stresses:
                p = p[:2]

            # Add space if last one isn't one.
            if p == space and len(ps) > 0 and ps[-1] != space:
                ps.append(p)

            # Add next phoneme
            if (p.isalnum() or p == "'") and p in labels:
                ps.append(p)

            # Add punct and remove space if needed
            if (p in self.PUNCT) and self.punct:
                if not self.spaces and len(ps) > 0 and ps[-1] == space:
                    ps.pop()
                ps.append(p)

        # Remove trailing spaces
        while ps[-1] == space:
            ps.pop()

        return [self._label2id[p] for p in ps]
