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

import os
import re
from typing import Dict, List, Optional, Union

import sentencepiece

from nemo.collections.common.parts.utils import if_exist
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.common.tokenizers.tokenizer_utils import MODEL_SPECIAL_TOKENS, TOKENIZERS
from nemo.utils import logging

__all__ = ['SentencePieceTokenizer', 'create_spt_model']


class SentencePieceTokenizer(TokenizerSpec):
    '''
    Sentencepiecetokenizer https://github.com/google/sentencepiece.
    '''

    def __init__(self, model_path: str, special_tokens: Optional[Union[Dict[str, str], List[str]]] = None):
        """
        Args:
            model_path: path to sentence piece tokenizer model. To create the model use create_spt_model()
            special_tokens: either list of special tokens or dictionary of token name to token value
        """
        self.tokenizer = sentencepiece.SentencePieceProcessor()
        self.tokenizer.Load(model_path)
        # without special tokens
        self.original_vocab_size = self.tokenizer.get_piece_size()
        self.vocab_size = self.tokenizer.get_piece_size()
        self.special_token_to_id = {}
        self.id_to_special_token = {}
        if special_tokens:
            self.add_special_tokens(special_tokens)

    def text_to_tokens(self, text):
        tokens = []
        idx = 0
        last_idx = 0

        while 1:
            indices = {}

            for token in self.special_token_to_id:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue

            if len(indices) == 0:
                break

            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            tokens.extend(self.tokenizer.encode_as_pieces(text[idx:next_idx]))
            tokens.append(next_token)
            idx = next_idx + len(next_token)

        tokens.extend(self.tokenizer.encode_as_pieces(text[idx:]))
        return tokens

    def text_to_ids(self, text):
        ids = []
        idx = 0
        last_idx = 0

        while 1:
            indices = {}

            for token in self.special_token_to_id:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue

            if len(indices) == 0:
                break

            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            ids.extend(self.tokenizer.encode_as_ids(text[idx:next_idx]))
            ids.append(self.special_token_to_id[next_token])
            idx = next_idx + len(next_token)

        ids.extend(self.tokenizer.encode_as_ids(text[idx:]))
        return ids

    def tokens_to_text(self, tokens):
        return self.tokenizer.decode_pieces(tokens)

    def ids_to_text(self, ids):
        text = ""
        last_i = 0

        for i, id in enumerate(ids):
            if id in self.id_to_special_token:
                text += self.tokenizer.decode_ids(ids[last_i:i]) + " "
                text += self.id_to_special_token[id] + " "
                last_i = i + 1

        text += self.tokenizer.decode_ids(ids[last_i:])
        return text.strip()

    def tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            tokens = [tokens]
        ids = []
        for token in tokens:
            ids.append(self.token_to_id(token))
        return ids

    def token_to_id(self, token):
        if token in self.special_token_to_id:
            return self.special_token_to_id[token]
        return self.tokenizer.piece_to_id(token)

    def ids_to_tokens(self, ids):
        tokens = []
        for id in ids:
            if id >= self.original_vocab_size:
                tokens.append(self.id_to_special_token[id])
            else:
                tokens.append(self.tokenizer.id_to_piece(id))
        return tokens

    def add_special_tokens(self, special_tokens):
        if isinstance(special_tokens, list):
            for token in special_tokens:
                if (
                    self.tokenizer.piece_to_id(token) == self.tokenizer.unk_id()
                    and token not in self.special_token_to_id
                ):
                    self.special_token_to_id[token] = self.vocab_size
                    self.id_to_special_token[self.vocab_size] = token
                    self.vocab_size += 1
        elif isinstance(special_tokens, dict):
            for token_name, token in special_tokens.items():
                setattr(self, token_name, token)
                if (
                    self.tokenizer.piece_to_id(token) == self.tokenizer.unk_id()
                    and token not in self.special_token_to_id
                ):
                    self.special_token_to_id[token] = self.vocab_size
                    self.id_to_special_token[self.vocab_size] = token
                    self.vocab_size += 1

    @property
    def pad_id(self):
        return self.tokens_to_ids([self.pad_token])[0]

    @property
    def bos_id(self):
        return self.tokens_to_ids([self.bos_token])[0]

    @property
    def eos_id(self):
        return self.tokens_to_ids([self.eos_token])[0]

    @property
    def sep_id(self):
        return self.tokens_to_ids([self.sep_token])[0]

    @property
    def cls_id(self):
        return self.tokens_to_ids([self.cls_token])[0]


def create_spt_model(
    data_file: str,
    vocab_size: int,
    sample_size: int,
    special_tokens: Optional[Union[Dict[str, str], List[str]]],
    do_lower_case: bool,
    output_dir: Optional[str] = None,
):
    """
    Creates sentence piece tokenizer model from data file.
    Args:
        data_file: data file
        vocab_size: vocabulary size
        sample_size: maximum size of sentences the trainer loads
        special_tokens: either list of special tokens or dictionary of token name to token value
        do_lower_case: if text should be lower cased before tokenizer model is created
        output_dir: folder to save created tokenizer model. If not specified will store model at data_file/../spt folder
    """

    if not data_file or not os.path.exists(data_file):
        raise ValueError(f"data_file must be valid file path, but got {data_file}")
    data_dir = os.path.dirname(data_file)
    if special_tokens:
        if isinstance(special_tokens, list):
            special_tokens = list(set(special_tokens))
        elif isinstance(special_tokens, dict):
            special_tokens = list(set(special_tokens.values()))
        vocab = special_tokens[:]
    else:
        vocab = []
    if not output_dir:
        output_dir = f'{data_dir}/spt'
    if if_exist(output_dir, ['tokenizer.model']):
        logging.info(f"tokenizer model {output_dir}/tokenizer.model already exists")
        return f'{output_dir}/tokenizer.model', f'{output_dir}/vocab.txt'
    logging.info(f'Processing {data_file} and store at {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    cmd = (
        f"--input={data_file} --model_prefix={output_dir}/tokenizer "
        f"--vocab_size={vocab_size - len(vocab)} "
        f"--shuffle_input_sentence=true --hard_vocab_limit=false "
        f"--bos_id=-1 --eos_id=-1"
    )
    if do_lower_case:
        cmd += " --normalization_rule_name=nmt_nfkc_cf"

    if sample_size > 0:
        cmd += f" --input_sentence_size={sample_size}"

    sentencepiece.SentencePieceTrainer.Train(cmd)

    # Add BERT control symbols
    tokens = []

    with open(f"{output_dir}/tokenizer.vocab", "r") as f:
        f.readline()  # skip first <unk> token

        # Read tokens from each line and parse for vocab
        for line in f:
            piece = line.split("\t")[0]
            token = piece[1:] if piece.startswith("▁") else f"##{piece}"

            if len(token) > 0:
                tokens.append(token)
            else:
                tokens.append(piece[0])

    vocab.extend(tokens)

    # Save vocabulary to output file
    vocab_file = f'{output_dir}/vocab.txt'
    with open(vocab_file, "w") as f:
        for token in vocab:
            f.write(f"{token}\n".format())
    return f'{output_dir}/tokenizer.model', vocab_file
