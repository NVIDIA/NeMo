# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


import re
import unicodedata
from builtins import str as unicode

# Derived from LJSpeech
_synoglyphs = {
    "'": ['’'],
    '"': ['”', '“'],
}
SYNOGLYPH2ASCII = {g: asc for asc, glyphs in _synoglyphs.items() for g in glyphs}

# Example of parsing by groups via _WORDS_RE.
# Groups:
# 1st group -- valid english words,
# 2nd group -- any substring starts from | to | (mustn't be nested), useful when you want to leave sequence unchanged,
# 3rd group -- punctuation marks.
# Text (first line) and mask of groups for every char (second line).
# config file must contain |EY1 EY1|, B, C, D, E, F, and G.
# 111111311113111131111111322222222233133133133133133111313
_WORDS_RE = re.compile("([a-zA-Z]+(?:[a-zA-Z-']*[a-zA-Z]+)*)|(\|[^|]*\|)|([^a-zA-Z|]+)")


def english_text_preprocessing(text, lower=True):
    text = unicode(text)
    text = ''.join(char for char in unicodedata.normalize('NFD', text) if unicodedata.category(char) != 'Mn')
    text = ''.join(char if char not in SYNOGLYPH2ASCII else SYNOGLYPH2ASCII[char] for char in text)

    if lower:
        text = text.lower()

    return text


def english_word_tokenize(text):
    """
    Convert text (str) to List[Tuple[Union[str, List[str]], bool]] where every tuple denotes word representation and flag whether to leave unchanged or not.
    Word can be one of: valid english word, any substring starts from | to | (unchangeable word) or punctuation marks.
    This function expects that unchangeable word is carefully divided by spaces (e.g. HH AH L OW).
    Unchangeable word will be splitted by space and represented as List[str], other cases are represented as str.
    """
    words = _WORDS_RE.findall(text)
    result = []
    for word in words:
        maybe_word, maybe_without_changes, maybe_punct = word

        if maybe_word != '':
            without_changes = False
            result.append((maybe_word.lower(), without_changes))
        elif maybe_punct != '':
            without_changes = False
            result.append((maybe_punct, without_changes))
        elif maybe_without_changes != '':
            without_changes = True
            result.append((maybe_without_changes[1:-1].split(" "), without_changes))
    return result
