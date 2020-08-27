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

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

__all__ = ['CharTokenizer']


class CharTokenizer(TokenizerSpec):
    def __init__(
        self,
        vocab_file: str,
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
    ):
        """
        Args:
            vocab_file: path to file with vocabulary which consists
                of characters separated by \n
            bos_token: the beginning of sequence token
            eos_token: the end of sequence token
            pad_token: token to use for padding
            unk_token: token to use for unknown tokens
        """

        vocab_list = open(vocab_file, "r").readlines()
        self.vocab = {vocab_list[i].strip(): i for i in range(len(vocab_list))}

        special_tokens_dict = {
            "bos_token": bos_token,
            "eos_token": eos_token,
            "pad_token": pad_token,
            "unk_token": unk_token,
        }

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
