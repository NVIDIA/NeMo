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

# MIT License
#
# Copyright (c) 2020 Jaehyeon Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
from typing import List, Optional

import inflect
from unidecode import unidecode


class CMUDict:
    def __init__(self, file_or_path, keep_ambiguous=True):
        """
        Thin wrapper around CMUDict data. http://www.speech.cs.cmu.edu/cgi-bin/cmudict
        Args:
            file_or_path: file or path to cmu dictionary
            keep_ambiguous: keep entries with multiple possible pronunciations
        """

        # fmt: off
        self.valid_symbols = [
            'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
            'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
            'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
            'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
            'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
            'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
            'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
        ]
        # fmt: on

        self._valid_symbol_set = set(self.valid_symbols)

        if isinstance(file_or_path, str):
            with open(file_or_path, encoding="latin-1") as f:
                entries = self._parse_cmudict(f)
        else:
            entries = self._parse_cmudict(file_or_path)
        if not keep_ambiguous:
            entries = {word: pron for word, pron in entries.items() if len(pron) == 1}
        self._entries = entries

    def __len__(self):
        return len(self._entries)

    def lookup(self, word):
        """Returns list of ARPAbet pronunciations of the given word."""
        return self._entries.get(word.upper())

    def _get_pronunciation(self, s):
        parts = s.strip().split(" ")
        for part in parts:
            if part not in self._valid_symbol_set:
                return None
        return " ".join(parts)

    def _parse_cmudict(self, file):
        _alt_re = re.compile(r"\([0-9]+\)")

        cmudict = {}
        for line in file:
            if len(line) and (line[0] >= "A" and line[0] <= "Z" or line[0] == "'"):
                parts = line.split("  ")
                word = re.sub(_alt_re, "", parts[0])
                pronunciation = self._get_pronunciation(parts[1])
                if pronunciation:
                    if word in cmudict:
                        cmudict[word].append(pronunciation)
                    else:
                        cmudict[word] = [pronunciation]
        return cmudict


class GlowTTSParser:
    def __init__(self, cmu_dict_path=None):
        """
        Parser for the glow tts model.
        Converts characters to phonemes where possible.
        Args:
            cmu_dict_path (str): Path to cmu dictionary
        """

        self.cmu_dict = None

        if cmu_dict_path:
            self.cmu_dict = CMUDict(cmu_dict_path)

        _pad = "_"
        _punctuation = "!'(),.:;? "
        _special = "-"
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

        self.symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters)

        if self.cmu_dict:
            # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
            _arpabet = ["@" + s for s in self.cmu_dict.valid_symbols]

            self.symbols = self.symbols + _arpabet

        # Mappings from symbol to numeric ID and vice versa:
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

        # Regular expression matching text enclosed in curly braces:
        self._curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

        # Regular expression matching whitespace:
        self._whitespace_re = re.compile(r"\s+")

        # List of (regular expression, replacement) pairs for abbreviations:
        self._abbreviations = [
            (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
            for x in [
                ("mrs", "missus"),
                ("mr", "mister"),
                ("dr", "doctor"),
                ("st", "saint"),
                ("co", "company"),
                ("jr", "junior"),
                ("maj", "major"),
                ("gen", "general"),
                ("drs", "doctors"),
                ("rev", "reverend"),
                ("lt", "lieutenant"),
                ("hon", "honorable"),
                ("sgt", "sergeant"),
                ("capt", "captain"),
                ("esq", "esquire"),
                ("ltd", "limited"),
                ("col", "colonel"),
                ("ft", "fort"),
            ]
        ]

        # List of other replacement pairs:
        self._replace = [
            (re.compile("\\b%s\\b" % x[0], re.IGNORECASE), x[1])
            for x in [
                ("_", "underscore"),
                ("src", "source"),
                ("dll", "d l l"),
                ("btw", "by the way"),
                ("http", "h t t p"),
                ("www", "w w w"),
                ("c\+\+", "c plus plus"),
            ]
        ]

        self._inflect = inflect.engine()
        self._comma_number_re = re.compile(r"([0-9][0-9,]+[0-9])")
        self._decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
        self._pounds_re = re.compile(r"£([0-9,]*[0-9]+)")
        self._dollars_re = re.compile(r"\$([0-9.,]*[0-9]+)")
        self._ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
        self._number_re = re.compile(r"[0-9]+")

    def __call__(self, text: str) -> Optional[List[int]]:
        return self.text_to_sequence(text, ["english_cleaners"], self.cmu_dict)

    def _expand_ordinal(self, m):
        return self._inflect.number_to_words(m.group(0))

    def _expand_number(self, m):
        num = int(m.group(0))
        if num > 1000 and num < 3000:
            if num == 2000:
                return "two thousand"
            elif num > 2000 and num < 2010:
                return "two thousand " + self._inflect.number_to_words(num % 100)
            elif num % 100 == 0:
                return self._inflect.number_to_words(num // 100) + " hundred"
            else:
                return self._inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
        else:
            return self._inflect.number_to_words(num, andword="")

    def expand_numbers(self, text):
        text = re.sub(self._comma_number_re, self._remove_commas, text)
        text = re.sub(self._pounds_re, r"\1 pounds", text)
        text = re.sub(self._dollars_re, self._expand_dollars, text)
        text = re.sub(self._decimal_number_re, self._expand_decimal_point, text)
        text = re.sub(self._ordinal_re, self._expand_ordinal, text)
        text = " ".join(re.split('(\d+)', text))
        text = re.sub(self._number_re, self._expand_number, text)
        return text

    def replace(self, text):
        for regex, replacement in self._abbreviations:
            text = re.sub(regex, replacement, text)
        for regex, replacement in self._replace:
            text = re.sub(regex, replacement, text)
        return text

    def lowercase(self, text):
        return text.lower()

    def collapse_whitespace(self, text):
        return re.sub(self._whitespace_re, " ", text)

    def convert_to_ascii(self, text):
        return unidecode(text)

    def basic_cleaners(self, text):
        """Basic pipeline that lowercases and collapses whitespace without transliteration."""
        text = self.lowercase(text)
        text = self.collapse_whitespace(text)
        return text

    def transliteration_cleaners(self, text):
        """Pipeline for non-English text that transliterates to ASCII."""
        text = self.convert_to_ascii(text)
        text = self.lowercase(text)
        text = self.collapse_whitespace(text)
        return text

    def english_cleaners(self, text):
        """Pipeline for English text, including number and abbreviation expansion."""
        text = self.convert_to_ascii(text)
        text = self.lowercase(text)
        text = self.expand_numbers(text)
        text = self.replace(text)
        text = self.collapse_whitespace(text)
        return text

    def get_arpabet(self, word, dictionary):
        word_arpabet = dictionary.lookup(word)
        if word_arpabet is not None:
            return "{" + word_arpabet[0] + "}"
        else:
            return word

    def text_to_sequence(self, text, cleaner_names, dictionary=None):
        """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

          The text can optionally have ARPAbet sequences enclosed in curly braces embedded
          in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

          Args:
            text: string to convert to a sequence
            cleaner_names: names of the cleaner functions to run the text through
            dictionary: arpabet class with arpabet dictionary

          Returns:
            List of integers corresponding to the symbols in the text
        """

        sequence = []

        space = self._symbols_to_sequence(" ")
        # Check for curly braces and treat their contents as ARPAbet:
        while len(text):
            m = self._curly_re.match(text)
            if not m:
                clean_text = self._clean_text(text, cleaner_names)
                if dictionary is not None:
                    clean_text = [self.get_arpabet(w, dictionary) for w in re.findall(r"[\w']+|[.,!?;]", clean_text)]
                    for i in range(len(clean_text)):
                        t = clean_text[i]
                        if t.startswith("{"):
                            sequence += self._arpabet_to_sequence(t[1:-1])
                        else:
                            sequence += self._symbols_to_sequence(t)
                        sequence += space
                else:
                    sequence += self._symbols_to_sequence(clean_text)
                break
            sequence += self._symbols_to_sequence(self._clean_text(m.group(1), cleaner_names))
            sequence += self._arpabet_to_sequence(m.group(2))
            text = m.group(3)

        # remove trailing space
        if dictionary is not None:
            sequence = sequence[:-1] if sequence[-1] == space[0] else sequence

        return sequence

    def sequence_to_text(self, sequence):
        """Converts a sequence of IDs back to a string"""
        result = ""
        for symbol_id in sequence:
            if symbol_id in self._id_to_symbol:
                s = self._id_to_symbol[symbol_id]
                # Enclose ARPAbet back in curly braces:
                if len(s) > 1 and s[0] == "@":
                    s = "{%s}" % s[1:]
                result += s
        return result.replace("}{", " ")

    def _clean_text(self, text, cleaner_names):
        for name in cleaner_names:
            cleaner = getattr(self, name, "english_cleaners")
            if not cleaner:
                raise Exception("Unknown cleaner: %s" % name)
            text = cleaner(text)
        return text

    def _symbols_to_sequence(self, symbols):
        return [self._symbol_to_id[s] for s in symbols if self._should_keep_symbol(s)]

    def _arpabet_to_sequence(self, text):
        return self._symbols_to_sequence(["@" + s for s in text.split()])

    def _should_keep_symbol(self, s):
        return s in self._symbol_to_id and s is not "_" and s is not "~"

    def _remove_commas(self, m):
        return m.group(1).replace(",", "")

    def _expand_decimal_point(self, m):
        return m.group(1).replace(".", " point ")

    def _expand_dollars(self, m):
        match = m.group(1)
        parts = match.split(".")
        if len(parts) > 2:
            return match + " dollars"  # Unexpected format
        dollars = int(parts[0]) if parts[0] else 0
        cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        if dollars and cents:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            cent_unit = "cent" if cents == 1 else "cents"
            return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
        elif dollars:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            return "%s %s" % (dollars, dollar_unit)
        elif cents:
            cent_unit = "cent" if cents == 1 else "cents"
            return "%s %s" % (cents, cent_unit)
        else:
            return "zero dollars"
