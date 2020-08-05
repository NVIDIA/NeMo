# Copyright 2020 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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

from transformers import GPT2Tokenizer

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging

__all__ = ['NemoGPT2Tokenizer']


class NemoGPT2Tokenizer(TokenizerSpec):
    def __init__(
        self,
        pretrained_model: str,
        bos_token: str = "<|endoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|pad|>",
        unk_token: str = "<|unk|>",
    ):
        """
        Args:
            pretrained_model: GPT2 pretrained model name from Hugging Face
            bos_token: the beginning of sequence token
            eos_token: the end of sequence token
            pad_token: token to use for padding
            unk_token: token to use for unknown tokens
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        self.vocab_size = self.tokenizer.vocab_size
        special_tokens_dict = {}
        if self.tokenizer.unk_token is None:
            self.tokenizer.unk_token = unk_token
            special_tokens_dict["unk_token"] = unk_token
        if self.tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = bos_token
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = eos_token
        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = pad_token
        self.add_special_tokens(special_tokens_dict)

    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        """
        Adds a dictionary of special tokens (eos, pad, cls...). If special tokens are NOT in the vocabulary, they are added
        to it (indexed starting from the last index of the current vocabulary).
        Args:
            special_tokens_dict: dict of string. Keys should be in the list of predefined special attributes:
                [``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``,
                ``additional_special_tokens``].
            Tokens are only added if they are not already in the vocabulary.
        Returns:
            Number of tokens added to the vocabulary.
        """
        num_tokens_added = self.tokenizer.add_special_tokens(special_tokens_dict)
        if num_tokens_added > 0:
            logging.info(f'{num_tokens_added} special tokens added, resize your model accordingly.')
        return num_tokens_added

    def text_to_tokens(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens):
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text

    def tokens_to_ids(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def ids_to_tokens(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens

    def text_to_ids(self, text):
        tokens = self.text_to_tokens(text)
        ids = self.tokens_to_ids(tokens)
        return ids

    def ids_to_text(self, ids):
        tokens = self.ids_to_tokens(ids)
        text = self.tokens_to_text(tokens)
        return text

    @property
    def pad_id(self):
        return self.tokens_to_ids([self.tokenizer.pad_token])[0]

    @property
    def bos_id(self):
        return self.tokens_to_ids([self.tokenizer.bos_token])[0]

    @property
    def eos_id(self):
        return self.tokens_to_ids([self.tokenizer.eos_token])[0]
