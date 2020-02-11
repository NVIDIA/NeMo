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

from nemo.collections.nlp.data.tokenizers.tokenizer_spec import TokenizerSpec

__all__ = ['CharTokenizer']


class CharTokenizer(TokenizerSpec):
    def __init__(self, vocab_path):

        vocab_list = open(vocab_path, "r").readlines()
        self.vocab = {vocab_list[i][0]: i for i in range(len(vocab_list))}
        for special_token in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            if special_token not in self.vocab:
                self.vocab[special_token] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.special_tokens = self.tokens_to_ids(["<PAD>", "<UNK>", "<BOS>", "<EOS>"])

    def text_to_tokens(self, text):
        token_candidates = [char for char in text]
        tokens = []
        for token in token_candidates:
            if token in self.vocab:
                tokens.append(token)
            else:
                tokens.append("<UNK>")
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
        return self.vocab["<PAD>"]

    @property
    def bos_id(self):
        return self.vocab["<BOS>"]

    @property
    def eos_id(self):
        return self.vocab["<EOS>"]
