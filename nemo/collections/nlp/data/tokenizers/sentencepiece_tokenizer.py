# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

import sentencepiece as spm

from nemo.collections.nlp.data.tokenizers.tokenizer_spec import TokenizerSpec

__all__ = ['SentencePieceTokenizer']


class SentencePieceTokenizer(TokenizerSpec):
    def __init__(self, model_path, special_tokens={}):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(model_path)
        # without special tokens
        self.original_vocab_size = self.tokenizer.get_piece_size()
        self.vocab_size = self.tokenizer.get_piece_size()
        self.special_token_to_id = {}
        self.id_to_special_token = {}
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

    def tokens_to_ids(self, tokens):
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
        return self.tokens_to_ids([getattr(self, 'pad_token')])[0]

    @property
    def bos_id(self):
        return self.tokens_to_ids([getattr(self, 'bos_token')])[0]

    @property
    def eos_id(self):
        return self.tokens_to_ids([getattr(self, 'eos_token')])[0]

    @property
    def sep_id(self):
        return self.tokens_to_ids([getattr(self, 'sep_token')])[0]

    @property
    def cls_id(self):
        return self.tokens_to_ids([getattr(self, 'cls_token')])[0]
