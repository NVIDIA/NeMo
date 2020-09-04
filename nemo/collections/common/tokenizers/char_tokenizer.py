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

from typing import Optional

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

__all__ = ['CharTokenizer']


class CharTokenizer(TokenizerSpec):
    "Tokenizes each character"

    def __init__(
        self,
        vocab_file: str,
        mask_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        cls_token: Optional[str] = None,
        unk_token: Optional[str] = None,
    ):
        """
        Args:
            vocab_file: path to file with vocabulary which consists
                of characters separated by \n
            mask_token: mask token 
            bos_token: the beginning of sequence token
            eos_token: the end of sequence token. Usually equal to sep_token
            pad_token: token to use for padding
            sep_token: token used for separating sequences
            cls_token: class token. Usually equal to bos_token
            unk_token: token to use for unknown tokens
        """

        vocab_list = open(vocab_file, "r").readlines()
        self.vocab = {vocab_list[i].strip(): i for i in range(len(vocab_list))}

        special_tokens_dict = {}
        if unk_token:
            special_tokens_dict["unk_token"] = unk_token
        if sep_token:
            special_tokens_dict["sep_token"] = sep_token
        if mask_token:
            special_tokens_dict["mask_token"] = mask_token
        if bos_token:
            special_tokens_dict["bos_token"] = bos_token
        if eos_token:
            special_tokens_dict["eos_token"] = eos_token
        if pad_token:
            special_tokens_dict["pad_token"] = pad_token
        if cls_token:
            special_tokens_dict["cls_token"] = cls_token

        self.add_special_tokens(special_tokens_dict)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.special_tokens = self.tokens_to_ids(special_tokens_dict.values())

    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        """
        Adds a dictionary of special tokens (eos, pad, cls...).
        If special tokens are NOT in the vocabulary, they are added
        to it (indexed starting from the last index of the current vocabulary).
        Args:
            special_tokens_dict: dict of special tokens
        """
        for token in special_tokens_dict:
            token_str = special_tokens_dict[token]
            if token_str not in self.vocab:
                self.vocab[token_str] = len(self.vocab)
            setattr(self, token, token_str)

    def text_to_tokens(self, text):
        token_candidates = [char for char in text]
        tokens = []
        for token in token_candidates:
            if token in self.vocab:
                tokens.append(token)
            else:
                tokens.append(self.unk_token)
        return tokens

    def tokens_to_text(self, tokens):
        return self.ids_to_text(self.tokens_to_ids(tokens))

    def text_to_ids(self, text):
        return [self.vocab[token] for token in self.text_to_tokens(text)]

    def ids_to_text(self, ids):
        ids_ = [id_ for id_ in ids if id_ not in self.special_tokens]
        return "".join(self.ids_to_tokens(ids_))

    def tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens]

    def token_to_id(self, token):
        return self.vocab[token]

    def ids_to_tokens(self, ids):
        return [self.inv_vocab[id] for id in ids]

    @property
    def pad_id(self):
        return self.vocab[self.pad_token]

    @property
    def bos_id(self):
        return self.vocab[self.bos_token]

    @property
    def eos_id(self):
        return self.vocab[self.eos_token]

    @property
    def unk_id(self):
        return self.vocab[self.unk_token]

    @property
    def mask_id(self):
        return self.vocab[self.mask_token]

    @property
    def sep_id(self):
        return self.vocab[self.sep_token]

    @property
    def cls_id(self):
        return self.vocab[self.cls_token]
