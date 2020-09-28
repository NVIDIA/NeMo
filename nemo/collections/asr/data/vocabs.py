"""Vocabularies for different datasets."""

import abc
import itertools
import string
from typing import List

import g2p_en

from nemo.collections.asr.parts import parsers


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
    SEP = ''

    def __init__(self, labels, *, pad=PAD, blank=BLANK, oov=OOV):
        super().__init__()

        labels = list(labels)
        self.pad, labels = len(labels), labels + [pad]  # Padding
        self.blank, labels = len(labels), labels + [blank]  # Reserved for blank from QN
        self.oov, labels = len(labels), labels + [oov]  # Out Of Vocabulary
        self.labels = labels

        self._util_ids = {self.pad, self.blank, self.oov}
        self._label2id = {l: i for i, l in enumerate(labels)}
        self._id2label = labels

    @abc.abstractmethod
    def encode(self, text: str) -> List[int]:
        """Turns str text into int tokens."""
        pass

    def decode(self, tokens: List[int]) -> str:
        """Turns ints tokens into str text."""
        return self.SEP.join(self._id2label[t] for t in tokens if t not in self._util_ids)


class Chars(Base):
    """Chars vocabulary."""

    def __init__(self, punct=True, spaces=False):
        labels = []
        self.space, labels = len(labels), labels + [' ']  # Space
        labels.extend(string.ascii_lowercase + "'")  # Apostrophe for saving "don't" and "Joe's"

        if punct:
            labels.extend(self.PUNCT)

        super().__init__(labels)

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

    _G2P = g2p_en.G2p()

    SEP = '|'  # To be able to distinguish between 2/3 letters codes.
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
        self, punct=True, stresses=False, spaces=True, *, space=' ', silence=None, oov=Base.OOV,
    ):
        labels = []
        self.space, labels = len(labels), labels + [space]  # Space
        if silence:
            self.silence, labels = len(labels), labels + [silence]  # Silence
        labels.extend(self.CONSONANTS)
        vowels = list(self.VOWELS)
        if stresses:
            vowels = [f'{p}{s}' for p, s in itertools.product(vowels, (0, 1, 2))]
        labels.extend(vowels)
        labels.append("'")  # Apostrophe

        if punct:
            labels.extend(self.PUNCT)

        super().__init__(labels, oov=oov)

        self.punct = punct
        self.stresses = stresses
        self.spaces = spaces

    def encode(self, text):
        """See base class."""
        ps, space = [], self.labels[self.space]

        for p in self._G2P(text):
            if len(p) == 3 and not self.stresses:
                p = p[:2]

            if p == space and ps[-1] != space:
                ps.append(p)

            if p.isalnum() or p == "'":
                ps.append(p)

            if p in self.PUNCT and self.punct:
                if not self.spaces and len(ps) and ps[-1] == space:
                    ps.pop()

                ps.append(p)

        if ps[-1] == space:
            ps.pop()

        return [self._label2id[p] for p in ps]


class MFA(Phonemes):
    """Montreal Forced Aligner set of phonemes."""

    SPACE, SILENCE, OOV = 'sp', 'sil', 'spn'

    def __init__(self):
        super().__init__(stresses=True, space=self.SPACE, silence=self.SILENCE, oov=self.OOV)

    def encode(self, text):
        """Split already parsed string of space delim phonemes codes into list of tokens."""
        return [self._label2id[p] for p in text.strip().split()]
