# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from collections import OrderedDict
from typing import List, Tuple

from nemo.collections.asr.inference.utils.constants import DEFAULT_SEMIOTIC_CLASS


# Compile regex pattern once at module level for better performance
TOKEN_PATTERN = re.compile(r'tokens \{.*?(?=tokens \{|$)', re.DOTALL)


def get_semiotic_class(tokens: List[OrderedDict]) -> str:
    """
    Returns the semiotic class of the given tokens.
    """
    return list(tokens[0]["tokens"].keys())[0]


def split_text(text: str, sep: str = " ") -> Tuple[List, int]:
    """
    Splits the text into words based on the separator.
    Args:
        text: (str) input text
        sep: (str) separator to split the text
    Returns:
        words: (List) list of words
        n_words: (int) number of words
    """
    cur_span = []
    words = []
    for idx, ch in enumerate(text):
        if ch == sep and len(cur_span) == 1:
            cur_span.append(idx)
            words.append(text[cur_span[0] : cur_span[1]])
            cur_span = []
        elif ch != sep and len(cur_span) == 0:
            cur_span.append(idx)

    if len(cur_span) > 0:
        cur_span.append(len(text))
        words.append(text[cur_span[0] : cur_span[1]])
    return words, len(words)


def find_tokens(text: str) -> List[str]:
    """
    Find the start and end positions of token blocks in the given text.
    Args:
        text: (str) input text containing token blocks
    Returns:
        token_blocks: (List[str]) list of token blocks
    """

    # Use compiled regex to find all token blocks in a single pass
    token_blocks = TOKEN_PATTERN.findall(text)

    # Strip whitespace from each block
    return [block.strip() for block in token_blocks]


def get_trivial_alignment(N: int, i_shift: int = 0, o_shift: int = 0) -> List[Tuple]:
    """
    Returns a trivial word alignment for N input words.
    Args:
        N: (int) number of input words
        i_shift: (int) input shift
        o_shift: (int) output shift
    Returns:
        (List) Returns a trivial word alignment
    """
    return [([i + i_shift], [i + o_shift], DEFAULT_SEMIOTIC_CLASS) for i in range(N)]


def fallback_to_trivial_alignment(
    input_words: List[str], i_shift: int = 0, o_shift: int = 0
) -> Tuple[List[str], List[str], List[Tuple]]:
    """
    Returns a trivial word alignment for the input words.
    Args:
        input_words: (List[str]) list of input words
        i_shift: (int) input shift
        o_shift: (int) output shift
    Returns:
        (Tuple) Returns a tuple of input words, output words, and a trivial word alignment
    """
    return input_words, input_words.copy(), get_trivial_alignment(N=len(input_words), i_shift=i_shift, o_shift=o_shift)
