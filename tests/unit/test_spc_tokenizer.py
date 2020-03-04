# ! /usr/bin/python
# -*- coding: utf-8 -*-

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

from unittest import TestCase

import pytest

import nemo.collections.nlp as nemo_nlp
from nemo.collections.nlp.data import SentencePieceTokenizer


class TestSPCTokenizer(TestCase):
    @pytest.mark.unit
    def test_add_special_tokens(self):
        tokenizer = SentencePieceTokenizer("./tests/data/m_common.model")
        special_tokens = nemo_nlp.data.tokenizers.MODEL_SPECIAL_TOKENS['bert']
        tokenizer.add_special_tokens(special_tokens)
        self.assertTrue(tokenizer.vocab_size == tokenizer.original_vocab_size + len(set(special_tokens.values())))

    @pytest.mark.unit
    def test_text_to_tokens(self):
        tokenizer = SentencePieceTokenizer("./tests/data/m_common.model")
        special_tokens = nemo_nlp.data.tokenizers.MODEL_SPECIAL_TOKENS['bert']
        tokenizer.add_special_tokens(special_tokens)

        text = "[CLS] a b c [MASK] e f [SEP] g h i [SEP]"
        tokens = tokenizer.text_to_tokens(text)

        self.assertTrue(len(tokens) == len(text.split()))
        self.assertTrue(tokens.count("[CLS]") == 1)
        self.assertTrue(tokens.count("[MASK]") == 1)
        self.assertTrue(tokens.count("[SEP]") == 2)

    @pytest.mark.unit
    def test_tokens_to_text(self):
        tokenizer = SentencePieceTokenizer("./tests/data/m_common.model")

        text = "[CLS] a b c [MASK] e f [SEP] g h i [SEP]"
        tokens = tokenizer.text_to_tokens(text)
        result = tokenizer.tokens_to_text(tokens)

        self.assertTrue(text == result)

    @pytest.mark.unit
    def test_text_to_ids(self):
        tokenizer = SentencePieceTokenizer("./tests/data/m_common.model")
        special_tokens = nemo_nlp.data.tokenizers.MODEL_SPECIAL_TOKENS['bert']
        tokenizer.add_special_tokens(special_tokens)

        text = "[CLS] a b c [MASK] e f [SEP] g h i [SEP]"
        ids = tokenizer.text_to_ids(text)

        self.assertTrue(len(ids) == len(text.split()))
        self.assertTrue(ids.count(tokenizer.token_to_id("[CLS]")) == 1)
        self.assertTrue(ids.count(tokenizer.token_to_id("[MASK]")) == 1)
        self.assertTrue(ids.count(tokenizer.token_to_id("[SEP]")) == 2)

    @pytest.mark.unit
    def test_ids_to_text(self):
        tokenizer = SentencePieceTokenizer("./tests/data/m_common.model")
        special_tokens = nemo_nlp.data.tokenizers.MODEL_SPECIAL_TOKENS['bert']
        tokenizer.add_special_tokens(special_tokens)

        text = "[CLS] a b c [MASK] e f [SEP] g h i [SEP]"
        ids = tokenizer.text_to_ids(text)
        result = tokenizer.ids_to_text(ids)

        self.assertTrue(text == result)

    @pytest.mark.unit
    def test_tokens_to_ids(self):
        tokenizer = SentencePieceTokenizer("./tests/data/m_common.model")
        special_tokens = nemo_nlp.data.tokenizers.MODEL_SPECIAL_TOKENS['bert']
        tokenizer.add_special_tokens(special_tokens)

        text = "[CLS] a b c [MASK] e f [SEP] g h i [SEP]"
        tokens = tokenizer.text_to_tokens(text)
        ids = tokenizer.tokens_to_ids(tokens)

        self.assertTrue(len(ids) == len(tokens))
        self.assertTrue(ids.count(tokenizer.token_to_id("[CLS]")) == 1)
        self.assertTrue(ids.count(tokenizer.token_to_id("[MASK]")) == 1)
        self.assertTrue(ids.count(tokenizer.token_to_id("[SEP]")) == 2)

    @pytest.mark.unit
    def test_ids_to_tokens(self):
        tokenizer = SentencePieceTokenizer("./tests/data/m_common.model")
        special_tokens = nemo_nlp.data.tokenizers.MODEL_SPECIAL_TOKENS['bert']
        tokenizer.add_special_tokens(special_tokens)

        text = "[CLS] a b c [MASK] e f [SEP] g h i [SEP]"
        tokens = tokenizer.text_to_tokens(text)
        ids = tokenizer.tokens_to_ids(tokens)
        result = tokenizer.ids_to_tokens(ids)

        self.assertTrue(len(result) == len(tokens))

        for i in range(len(result)):
            self.assertTrue(result[i] == tokens[i])
