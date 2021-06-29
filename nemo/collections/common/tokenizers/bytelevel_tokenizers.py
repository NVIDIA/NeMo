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

import re
from pathlib import Path
from typing import List

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

__all__ = ['ByteLevelProcessor', 'ByteLevelTokenizer']


class ByteLevelProcessor:
    """
    A very basic tokenization and detokenization class for use with byte-level
    tokenization.
    """

    def detokenize(self, tokens: List[str]) -> str:
        return ' '.join(tokens)

    def tokenize(self, text) -> str:
        return text

    def normalize(self, text) -> str:
        return text


class ByteLevelTokenizer(TokenizerSpec):
    def __init__(self):
        self.vocab_size = 259
        self.special_tokens = [self.bos_id, self.eos_id, self.pad_id]

    # no distinction between tokens and ids.
    def text_to_tokens(self, text):
        return self.text_to_ids(text)

    def tokens_to_text(self, tokens):
        return self.ids_to_text(tokens)

    def text_to_ids(self, text):
        return list(text.encode('utf-8'))

    def ids_to_text(self, ids):
        # remove special tokens.
        ids = [x for x in ids if x < 256]
        return bytes(ids).decode('utf-8', errors='ignore').rstrip()

    def tokens_to_ids(self, tokens):
        return tokens

    def ids_to_tokens(self, ids):
        return ids

    @property
    def pad_id(self):
        return 256

    @property
    def bos_id(self):
        return 257

    @property
    def eos_id(self):
        return 258

    @property
    def unk_id(self):
        return 259  # unused
