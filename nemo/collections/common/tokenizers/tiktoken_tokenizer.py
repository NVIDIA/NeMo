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
import re
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

    def text_to_tokens(self, text: str) -> List[str]:
        return self.ids_to_tokens(self.text_to_ids(text))

    def tokens_to_text(
        self,
        tokens: List[str],
        skip_special_tokens: bool = False,
        skip_bos_token: bool = True,
        skip_eos_token: bool = True,
    ) -> str:
        return self.ids_to_text(
            self.tokens_to_ids(tokens),
            skip_special_tokens=skip_special_tokens,
            skip_bos_token=skip_bos_token,
            skip_eos_token=skip_eos_token,
        )

    def token_to_id(self, token):
        token_ids = self.tokens_to_ids([token])
        if len(token_ids) != 1:
            raise ValueError(f"Token '{token}' should correspond to exactly one ID, but got {token_ids}")
        return token_ids[0]

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
        chunks = []
        for id_ in ids:
            if id_ < self.num_special_tokens:
                if chunks:
                    # Decode the chunk and append resulting tokens
                    tokens += self._ids_to_tokens_core([t - self.num_special_tokens for t in chunks])
                    chunks = []
                # Add the special token directly
                tokens.append(self.special_tokens[id_])
            else:
                # Add to current chunk
                chunks.append(id_)
        if chunks:
            # Decode any remaining chunk
            tokens += self._ids_to_tokens_core([t - self.num_special_tokens for t in chunks])
        # If there is no valid token, we return a single empty token. This is because some code in NeMo
        # expects this method to always return at least one token.
        return tokens if tokens else [""]

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

    def ids_to_text(
        self,
        ids: List[int],
        skip_special_tokens: bool = False,
        skip_bos_token: bool = True,
        skip_eos_token: bool = True,
    ) -> str:
        result = []
        chunks = []
        for id_ in ids:
            if id_ < self.num_special_tokens:
                if chunks:
                    result.append(self.tokenizer.decode([t - self.num_special_tokens for t in chunks]))
                    chunks = []
                skip = (
                    skip_bos_token
                    if id_ == self.bos_id
                    else (skip_eos_token if id_ == self.eos_id else skip_special_tokens)
                )
                if not skip:
                    result.append(self.special_tokens[id_])
            else:
                chunks.append(id_)
        if chunks:
            result.append(self.tokenizer.decode([t - self.num_special_tokens for t in chunks]))
        return ''.join(result)

    def _ids_to_tokens_core(self, ids: List[int]) -> List[str]:
        """
        Core implementation of `ids_to_tokens()`.

        Here the input `ids` are assume to be already shifted by `num_special_tokens` (and thus
        not to contain any special token).
        """
        tokens_bytes = self.tokenizer.decode_tokens_bytes(ids)
        tokens = []
        idx = 0
        while idx < len(tokens_bytes):
            try:
                # The most common case is typically that we can decode the token ID directly
                # into a UTF-8 string.
                tokens.append(tokens_bytes[idx].decode("utf-8", errors="strict"))
                idx += 1
            except UnicodeDecodeError:
                # If this fails, it may mean several things:
                #   1. (Most likely) This token spans multiple token IDs due to multi-byte UTF-8 encoding.
                #   2. (Less likely) somehow the input IDs lead to "invalid" UTF-8.
                # Although it would be possible to explicitly check byte values for multi-byte UTF-8 encoding
                # being spread across multiple token IDs, here we choose a simpler (albeit less efficient in
                # at least some situations) implementation where we incrementally add bytes obtained from other
                # token IDs until a valid decoding is obtained.
                # If no valid decoding can be obtained, this token is skipped.
                start_idx = idx
                while idx < len(tokens_bytes):
                    idx += 1
                    candidate_bytes = b"".join(tokens_bytes[start_idx:idx])
                    try:
                        tokens.append(candidate_bytes.decode("utf-8", errors="strict"))
                        break
                    except UnicodeDecodeError:
                        continue
                else:  # NB: this is the `else:` from `while`, not from `try...except`
                    # Failed to find a valid candidate => skip current token.
                    idx = start_idx + 1
        return tokens

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
