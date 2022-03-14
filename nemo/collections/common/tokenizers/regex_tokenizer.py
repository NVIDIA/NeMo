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


import os
import re
from typing import Optional, List
import pandas as pd

from nemo.utils import logging
from nemo.collections.common.tokenizers.char_tokenizer import TokenizerSpec

__all__ = ['RegExTokenizer']

DEFAULT_MASK_TOKEN = '<MASK>'
DEFAULT_BOS_TOKEN = '^'
DEFAULT_EOS_TOKEN = '&'
DEFAULT_PAD_TOKEN = '<PAD>'
DEFAULT_SEP_TOKEN = '<SEP>'
DEFAULT_UNK_TOKEN = '?'

class RegExTokenizer(TokenizerSpec):
    "A regular expression-based tokenizer at word boundary"

    def __init__(
        self,
        vocab_file: str,
        regex: str,
        mask_token: Optional[str] = DEFAULT_MASK_TOKEN,
        bos_token: Optional[str] = DEFAULT_BOS_TOKEN,
        eos_token: Optional[str] = DEFAULT_EOS_TOKEN,
        pad_token: Optional[str] = DEFAULT_PAD_TOKEN,
        sep_token: Optional[str] = DEFAULT_SEP_TOKEN,
        unk_token: Optional[str] = DEFAULT_UNK_TOKEN,
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
        self.regex = regex
        self.mask_token = mask_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.sep_token = sep_token
        self.unk_token = unk_token

        if not vocab_file or not os.path.exists(vocab_file):
            raise ValueError(f"Vocab file: {vocab_file} is invalid")
        self.vocab_file = vocab_file
        self.load_vocab()

        # Computed attributes
        self._compiled_regex = None
        self._compile_regex()

        ## Cache data/attributes required for tokenization
        self._unk_id = self.vocab.get(unk_token, DEFAULT_UNK_TOKEN)
        self._decode_vocab = {i: t for t, i in self.vocab.items()}

    def _compile_regex(self):
        regex_string = r"("
        regex_string += self.regex + r"|"
        regex_string += r".)"
        self._compiled_regex = re.compile(regex_string)

    def text_to_tokens(self, text):
        # Begin token
        tokens = [self.bos_token]
        tokens.extend(self._compiled_regex.findall(text))
        # End token
        tokens.append(self.eos_token)

        return tokens

    def tokens_to_text(self, tokens):
        tokens_list = []
        for token in tokens:
            if token[0] == self.bos_token:
                token = token[1:]

            # Remove end token and the following values
            if self.eos_token in token:
                eos_idx = token.index(self.eos_token)
                token = token[:eos_idx]

            tokens_list.append(token)

        text = ["".join(tokens) for tokens in tokens_list]
        return text

    def token_to_ids(self, tokens):
        ids_list = []
        for token in tokens:
            ids_list.append(self.vocab.get(token, self._unk_id))
        return ids_list

    def tokens_to_ids(self, token_data):
        if isinstance(token_data, str):
            token_data = [token_data]

        ids_list = []
        for tokens in token_data:
            ids = self.token_to_ids(tokens)
            ids_list.append(ids)
        return ids_list

    def ids_to_tokens(self, ids):
        tokens_list = []
        for ids in ids:
            for token_id in ids:
                token = self._decode_vocab.get(token_id)
                if token is None:
                    raise ValueError(f"Token id {token_id} is not recognised")

            tokens = [self._decode_vocab.get(token_id) for token_id in ids]
            tokens_list.append(tokens)

        return tokens_list

    def text_to_ids(self, text):
        tokens = self.text_to_tokens(text)
        tokens = [tokens]
        return self.tokens_to_ids(tokens)[0]

    def ids_to_text(self, ids):
        tokens = self.ids_to_tokens(ids)
        return self.tokens_to_text(tokens)

    def load_vocab(self):
        vocab = {}
        with open(self.vocab_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    vocab[line] = len(vocab)
        self.vocab = vocab

    @staticmethod
    def create_vocab(data_csv_file, vocab_file, smiles_col="smiles"):
        # NOTE this has to be run on each CSV file
        if not os.path.exists(data_csv_file):
            raise ValueError(f"Data file: {data_csv_file} is invalid")

        # Create empty vocab file
        if not os.path.exists(vocab_file):
            fp = open(vocab_file, 'w')
            fp.close()

        df = pd.read_csv(data_csv_file)
        tokenizer = RegExTokenizer(vocab_file=vocab_file)

        vocab = {
            DEFAULT_PAD_TOKEN : 0, # pad_token
            DEFAULT_UNK_TOKEN : 1, # unk_token
            DEFAULT_BOS_TOKEN : 2, # begin_token
            DEFAULT_EOS_TOKEN : 3, # end_token
            DEFAULT_MASK_TOKEN: 4, # mask_token
            DEFAULT_SEP_TOKEN : 5  # sep_token
        }
        for smiles in df[smiles_col]:
            tokens = tokenizer.text_to_tokens(smiles)
            logging.debug(f"SMILES: {smiles}, Tokens: {tokens}")
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

        vocab = sorted(vocab.items(), key=lambda k_v: k_v[1])
        logging.debug(f"Vocab: {vocab}")

        with open(vocab_file, 'w') as fp:
            for token in vocab:
                fp.write(f"{token[0]}\n")
#=============================================================================#
# HERE
#=============================================================================#

from typing import Optional

from nemo.collections.common.tokenizers.char_tokenizer import CharTokenizer

__all__ = ['WordTokenizer']


class WordTokenizer(CharTokenizer):
    "Tokenizes at word boundary"

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

        super().__init__(
            vocab_file=vocab_file,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
        )

    def text_to_tokens(self, text):
        token_candidates = text.strip().split()
        tokens = []
        for token in token_candidates:
            if token in self.vocab:
                tokens.append(token)
            else:
                tokens.append(self.unk_token)
        return tokens

    def ids_to_text(self, ids):
        ids_ = [id_ for id_ in ids if id_ not in self.special_tokens]
        return " ".join(self.ids_to_tokens(ids_))
