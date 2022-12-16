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

from typing import Dict, List, Optional, Union

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
    def __init__(self, special_tokens: Optional[Union[Dict[str, str], List[str]]] = None):
        self.vocab_size = 259
        self.special_start = 256
        self.special_token_to_id = {
            self.pad_id: self.pad_id,
            self.bos_id: self.bos_id,
            self.eos_id: self.eos_id,
        }
        special_tokens = {} if special_tokens is None else special_tokens
        for tok in special_tokens:
            self.special_start -= 1
            self.special_token_to_id[tok] = self.special_start

        self.id_to_special_token = {v: k for k, v in self.special_token_to_id.items()}

    # no distinction between tokens and ids.
    def text_to_tokens(self, text):
        return self.text_to_ids(text)

    def tokens_to_text(self, tokens):
        return self.ids_to_text(tokens)

    def text_to_ids(self, text):
        return list(text.encode('utf-8'))

    def ids_to_text(self, ids):
        # remove special tokens.
        ids = [x for x in ids if x < self.special_start]
        return bytes(ids).decode('utf-8', errors='ignore').rstrip()

    def tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens]
        ids = []
        for token in tokens:
            ids.append(self.token_to_id(token))
        return ids

    def ids_to_tokens(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        tokens = []
        for id in ids:
            tokens.append(self.id_to_token(id))
        return tokens

    def token_to_id(self, token):
        if token in self.special_token_to_id:
            return self.special_token_to_id[token]
        else:
            return token

    def id_to_token(self, id):
        if id < self.special_start:
            return id
        else:
            return self.id_to_special_token[id]

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
