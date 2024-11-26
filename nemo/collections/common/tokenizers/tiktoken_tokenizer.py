# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import base64
import json
import os
import regex as re
from pathlib import Path
from typing import Dict, List, Optional

try:
    import tiktoken
except ImportError:
    pass

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

__all__ = ['TiktokenTokenizer']


def reload_mergeable_ranks(
    path: str,
    max_vocab: Optional[int] = None,
) -> Dict[bytes, int]:
    """
    Reload the tokenizer JSON file and convert it to Tiktoken format.
    """
    assert path.endswith(".json")

    # reload vocab
    with open(path, "r") as f:
        vocab = json.load(f)
    assert isinstance(vocab, list)
    print(f"Vocab size: {len(vocab)}")
    if max_vocab is not None:
        vocab = vocab[:max_vocab]
        print(f"Cutting vocab to first {len(vocab)} tokens.")

    # build ranks
    ranks: Dict[bytes, int] = {}
    for i, x in enumerate(vocab):
        assert x.keys() == {"rank", "token_bytes", "token_str"}
        assert x["rank"] == i
        merge = base64.b64decode(x["token_bytes"])
        assert i >= 256 or merge == bytes([i])
        ranks[merge] = x["rank"]

    # sanity check
    assert len(ranks) == len(vocab)
    assert set(ranks.values()) == set(range(len(ranks)))

    return ranks


PATTERN_TIKTOKEN = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
DEFAULT_TIKTOKEN_MAX_VOCAB = 2**17  # 131072
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>"]
SPECIAL_TOKEN_TEMPLATE = "<SPECIAL_{id}>"

class TiktokenTokenizer(TokenizerSpec):
    """
    TiktokenTokenizer https://github.com/openai/tiktoken.

    Args:
        model_path: path to tokenizer vocabulary
        num_special_tokens: number of special tokens to generate
        special_tokens: template for user-defined special tokens
        pattern: Regex pattern to split the text
    """

    def __init__(
        self,
        vocab_file: str,
        pattern: str = PATTERN_TIKTOKEN,
        vocab_size: int = DEFAULT_TIKTOKEN_MAX_VOCAB,  # 131072
        num_special_tokens: int = 1000,
        special_tokens: Optional[List[str]] = None,
    ):
        if not vocab_file or not os.path.exists(vocab_file):
            raise ValueError(f"vocab_file: {vocab_file} is invalid")

        if special_tokens is None:
            special_tokens = SPECIAL_TOKENS.copy()

        assert len(special_tokens) == len(set(special_tokens)), f"Special tokens should be unique: {special_tokens}"
        assert len(special_tokens) <= num_special_tokens < vocab_size
        assert set(SPECIAL_TOKENS) <= set(special_tokens), f"Custom special tokens should include {SPECIAL_TOKENS}"

        self._unk_id = special_tokens.index("<unk>")
        self._bos_id = special_tokens.index("<s>")
        self._eos_id = special_tokens.index("</s>")

        self._vocab_size = vocab_size

        self.num_special_tokens = num_special_tokens
        special_filler = [SPECIAL_TOKEN_TEMPLATE.format(id=i) for i in range(len(special_tokens), num_special_tokens)]
        if special_filler:
            print(f"Adding special tokens {special_filler[0]}, ..., {special_filler[-1]}")
        self.special_tokens = special_tokens + special_filler
        assert len(set(self.special_tokens)) == len(self.special_tokens) == num_special_tokens, self.special_tokens
        self.inner_vocab_size = vocab_size - num_special_tokens

        # reload vocab
        self.token2id = reload_mergeable_ranks(vocab_file, max_vocab=self.inner_vocab_size)
        self.id2token = {v: k for k, v in self.token2id.items()}
        assert set(range(self.inner_vocab_size)) == set(self.id2token.keys())

        self.shifted_id2token = {i: tok for i, tok in enumerate(self.special_tokens)}
        for key, value in self.id2token.items():
            self.shifted_id2token[key + self.num_special_tokens] = value

        self.tokenizer = tiktoken.Encoding(
            name=Path(vocab_file).parent.name,
            pat_str=pattern,
            mergeable_ranks=self.token2id,
            special_tokens={},  # special tokens are handled manually
        )
        
        # Compile the tokenizer pattern for later use
        self.pattern = re.compile(pattern)

    def text_to_tokens(self, text: str) -> List[str]:
        return self.ids_to_tokens(self.text_to_ids(text))

    def tokens_to_text(self, tokens: List[str]) -> str:
        return ''.join(tokens)

    def token_to_id(self, token):
        token_str = token.decode('utf-8', errors='replace') if isinstance(token, bytes) else token
        if token_str in self.special_tokens:
            return self.special_tokens.index(token_str)
        else:
            token_ids = self.tokenizer.encode(token_str)
            if len(token_ids) != 1:
                raise ValueError(f"Token '{token_str}' should correspond to exactly one ID, but got {token_ids}")
            return token_ids[0] + self.num_special_tokens

    def tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            token_str = token.decode('utf-8', errors='replace') if isinstance(token, bytes) else token
            if token_str in self.special_tokens:
                ids.append(self.special_tokens.index(token_str))
            else:
                ids.extend([id + self.num_special_tokens for id in self.tokenizer.encode(token_str)])
        return ids

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        tokens = []
        current_ids = []
        for id_ in ids:
            if id_ < self.num_special_tokens:
                if current_ids:
                    adjusted_ids = [i - self.num_special_tokens for i in current_ids]
                    text, offsets = self.tokenizer.decode_with_offsets(adjusted_ids)
                    num_tokens = len(offsets)
                    for i in range(num_tokens):
                        start = offsets[i]
                        end = offsets[i + 1] if i + 1 < num_tokens else len(text)
                        token = text[start:end]
                        tokens.append(token)
                    current_ids = []
                tokens.append(self.special_tokens[id_])
            else:
                current_ids.append(id_)
        if current_ids:
            adjusted_ids = [i - self.num_special_tokens for i in current_ids]
            text, offsets = self.tokenizer.decode_with_offsets(adjusted_ids)
            num_tokens = len(offsets)
            for i in range(num_tokens):
                start = offsets[i]
                end = offsets[i + 1] if i + 1 < num_tokens else len(text)
                token = text[start:end]
                tokens.append(token)
        return tokens

    def text_to_ids(self, text: str) -> List[int]:
        ids = []
        special_token_pattern = SPECIAL_TOKEN_TEMPLATE.format(id=r"\d+")
        parts = re.split(f"({special_token_pattern}|<unk>|<s>|</s>)", text)
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.special_tokens.index(part))
            else:
                token_ids = self.tokenizer.encode(part)
                ids.extend([t + self.num_special_tokens for t in token_ids])
        return ids

    def ids_to_text(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        result = []
        chunks = []
        for id_ in ids:
            if id_ < self.num_special_tokens:
                if chunks:
                    result.append(self.tokenizer.decode([t - self.num_special_tokens for t in chunks]))
                    chunks = []
                if not skip_special_tokens:
                    result.append(self.special_tokens[id_])
            else:
                chunks.append(id_)
        if chunks:
            result.append(self.tokenizer.decode([t - self.num_special_tokens for t in chunks]))
        return ''.join(result)
    
    @property
    def bos_id(self):
        return self._bos_id

    @property
    def eos_id(self):
        return self._eos_id

    @property
    def unk_id(self):
        return self._unk_id

    @property
    def vocab(self):
        return self.token2id

    @property
    def decoder(self):
        return self.shifted_id2token

    @property
    def encoder(self):
        return self.vocab

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
