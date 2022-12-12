# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import json
import math
import os
import sys
from io import open
from typing import Dict, List, Tuple

import numpy as np
import regex as re
from numpy import ndarray

from nemo.utils import logging

__all__ = ["MusicNodes", "bytes_to_unicode"]


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    _chr = unichr if sys.version_info[0] == 2 else chr
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [_chr(n) for n in cs]
    return dict(zip(bs, cs))


class MusicNodes(object):
    """
    Modified for Tristan Behrens
    """

    def __init__(self, vocab_file, errors='replace', special_tokens='<|endoftext|>', max_len=None):
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = json.load(open(vocab_file, 'r'))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for
        # capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def __len__(self):
        return len(self.encoder) + len(self.special_tokens)

    def tokenize(self, text):
        """ Simplified Tokenizer for Music """
        tokens = text.split(' ')

        return tokens

    def convert_tokens_to_ids(self, tokens):
        """ Converts a sequence of tokens into ids using the vocab. """
        ids = []
        for token in tokens:
            ids.append(self.encoder.get(token, 0))
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this OpenAI GPT model ({} > {}). Running this"
                " sequence through the model will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """Converts a sequence of ids in tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.decoder[i])
        return tokens

    def encode(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text
