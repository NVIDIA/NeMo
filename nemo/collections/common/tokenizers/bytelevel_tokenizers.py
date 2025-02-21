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
        """
        Detokenize a list of tokens into a string.
        """
        return ' '.join(tokens)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string into a list of tokens.
        """
        return list(text)

    def normalize(self, text: str) -> str:
        """
        Normalize a string.
        """
        return text


class ByteLevelTokenizer(TokenizerSpec):
    """
    A byte-level tokenizer that encodes text as UTF-8 bytes with user control over the EOS, BOS, and PAD
        tokens as well as the vocabulary size and a mapping of other special tokens to their IDs.
    """

    def __init__(
        self,
        special_tokens: Optional[Union[Dict[str, str], List[str]]] = None,
        vocab_size: int = 512,
        _eos_id: int = 0,
        _pad_id: int = 1,
        _bos_id: int = None,
    ):
        """A byte-level tokenizer that encodes text as UTF-8 bytes.

        This tokenizer treats each byte as a token, with a default vocabulary size of 512 to accommodate
        UTF-8 byte values (0-255) plus special tokens. It can handle arbitrary text input by encoding
        it into bytes.

        Args:
            special_tokens: Dictionary or list of special tokens to add to the vocabulary.
                These tokens will be assigned IDs at the end of the vocabulary.
                Defaults to None.
            vocab_size: Size of the vocabulary, should be at least 256 to handle all byte values.
                Special tokens will be added after this size.
                Defaults to 512.
            _eos_id: ID to use for the end-of-sequence token.
                Defaults to 0.
            _pad_id: ID to use for the padding token.
                Defaults to 1.
            _bos_id: ID to use for the beginning-of-sequence token.
                Defaults to None.
        """
        self.vocab_size = vocab_size if special_tokens is None else vocab_size + len(special_tokens)
        self.special_start = vocab_size
        self._eos_id = _eos_id
        self._pad_id = _pad_id
        self._bos_id = _bos_id
        self.special_token_to_id = {
            self.pad_id: self.pad_id,
            self.bos_id: self.bos_id,
            self.eos_id: self.eos_id,
        }
        # Track special byte-tokens at end of vocabulary.
        self.vocab_size = vocab_size if special_tokens is None else vocab_size + len(special_tokens)
        self.special_start = self.vocab_size
        special_tokens = {} if special_tokens is None else special_tokens
        for tok in special_tokens:
            self.special_start -= 1
            self.special_token_to_id[tok] = self.special_start
        self.id_to_special_token = {v: k for k, v in self.special_token_to_id.items()}
