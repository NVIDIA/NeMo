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

import pytest
import torch

from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

MODEL_SPECIAL_TOKENS = {
    'unk_token': '[UNK]',
    'sep_token': '[SEP]',
    'pad_token': '[PAD]',
    'bos_token': '[CLS]',
    'mask_token': '[MASK]',
    'eos_token': '[SEP]',
    'cls_token': '[CLS]',
}


class TestSentencePieceTokenizerLegacy:
    model_name = "/m_common.model"

    @pytest.mark.unit
    def test_add_special_tokens(self, test_data_dir):
        tokenizer = SentencePieceTokenizer(test_data_dir + self.model_name, legacy=True)
        special_tokens = MODEL_SPECIAL_TOKENS
        tokenizer.add_special_tokens(special_tokens)
        assert tokenizer.vocab_size == tokenizer.original_vocab_size + len(set(special_tokens.values()))

    @pytest.mark.unit
    def test_text_to_tokens(self, test_data_dir):
        tokenizer = SentencePieceTokenizer(test_data_dir + self.model_name, legacy=True)
        special_tokens = MODEL_SPECIAL_TOKENS
        tokenizer.add_special_tokens(special_tokens)

        text = "[CLS] a b c [MASK] e f [SEP] g h i [SEP]"
        tokens = tokenizer.text_to_tokens(text)

        assert len(tokens) == len(text.split())
        assert tokens.count("[CLS]") == 1
        assert tokens.count("[MASK]") == 1
        assert tokens.count("[SEP]") == 2

    @pytest.mark.unit
    def test_tokens_to_text(self, test_data_dir):
        tokenizer = SentencePieceTokenizer(test_data_dir + self.model_name, legacy=True)

        text = "[CLS] a b c [MASK] e f [SEP] g h i [SEP]"
        tokens = tokenizer.text_to_tokens(text)
        result = tokenizer.tokens_to_text(tokens)

        assert text == result

    @pytest.mark.unit
    def test_text_to_ids(self, test_data_dir):
        tokenizer = SentencePieceTokenizer(test_data_dir + self.model_name, legacy=True)
        special_tokens = MODEL_SPECIAL_TOKENS
        tokenizer.add_special_tokens(special_tokens)

        text = "[CLS] a b c [MASK] e f [SEP] g h i [SEP]"
        ids = tokenizer.text_to_ids(text)

        assert len(ids) == len(text.split())
        assert ids.count(tokenizer.token_to_id("[CLS]")) == 1
        assert ids.count(tokenizer.token_to_id("[MASK]")) == 1
        assert ids.count(tokenizer.token_to_id("[SEP]")) == 2

    @pytest.mark.unit
    def test_ids_to_text(self, test_data_dir):
        tokenizer = SentencePieceTokenizer(test_data_dir + self.model_name, legacy=True)
        special_tokens = MODEL_SPECIAL_TOKENS
        tokenizer.add_special_tokens(special_tokens)

        text = "[CLS] a b c [MASK] e f [SEP] g h i [SEP]"
        ids = tokenizer.text_to_ids(text)
        result = tokenizer.ids_to_text(ids)

        assert text == result

    @pytest.mark.unit
    def test_tokens_to_ids(self, test_data_dir):
        tokenizer = SentencePieceTokenizer(test_data_dir + self.model_name, legacy=True)
        special_tokens = MODEL_SPECIAL_TOKENS
        tokenizer.add_special_tokens(special_tokens)

        text = "[CLS] a b c [MASK] e f [SEP] g h i [SEP]"
        tokens = tokenizer.text_to_tokens(text)
        ids = tokenizer.tokens_to_ids(tokens)

        assert len(ids) == len(tokens)
        assert ids.count(tokenizer.token_to_id("[CLS]")) == 1
        assert ids.count(tokenizer.token_to_id("[MASK]")) == 1
        assert ids.count(tokenizer.token_to_id("[SEP]")) == 2

    @pytest.mark.unit
    def test_ids_to_tokens(self, test_data_dir):
        tokenizer = SentencePieceTokenizer(test_data_dir + self.model_name, legacy=True)
        special_tokens = MODEL_SPECIAL_TOKENS
        tokenizer.add_special_tokens(special_tokens)

        text = "[CLS] a b c [MASK] e f [SEP] g h i [SEP]"
        tokens = tokenizer.text_to_tokens(text)
        ids = tokenizer.tokens_to_ids(tokens)
        result = tokenizer.ids_to_tokens(ids)

        assert len(result) == len(tokens)

        for i in range(len(result)):
            assert result[i] == tokens[i]


class TestSentencePieceTokenizer:
    model_name = "/m_new.model"

    @pytest.mark.unit
    def test_text_to_tokens(self, test_data_dir):
        tokenizer = SentencePieceTokenizer(test_data_dir + self.model_name)

        # <cls> is user_defined_symbol in the test tokenizer model
        # <unk>, <sep>, <s>, and </s> are control symbols
        text = "<cls> a b c <sep> e f g h i </s>"
        tokens = tokenizer.text_to_tokens(text)

        assert tokens.count("<cls>") == 1
        assert tokens.count("<sep>") == 0
        assert tokens.count("</s>") == 0

    @pytest.mark.unit
    def test_tokens_to_text(self, test_data_dir):
        tokenizer = SentencePieceTokenizer(test_data_dir + self.model_name)

        # <cls> is user_defined_symbol in the test tokenizer model
        text = "<cls> a b c e f g h i"
        tokens = tokenizer.text_to_tokens(text)
        result = tokenizer.tokens_to_text(tokens)

        assert text == result

    @pytest.mark.unit
    def test_text_to_ids(self, test_data_dir):
        tokenizer = SentencePieceTokenizer(test_data_dir + self.model_name)

        # <cls> is user_defined_symbol in the test tokenizer model
        # <unk>, <sep>, <s>, and </s> are control symbols
        text = "<cls> a b c <sep> e f g h i </s>"
        tokens = tokenizer.text_to_ids(text)

        assert tokens.count(tokenizer.token_to_id("<cls>")) == 1
        assert tokens.count(tokenizer.token_to_id("<sep>")) == 0
        assert tokens.count(tokenizer.token_to_id("</s>")) == 0

    @pytest.mark.unit
    def test_ids_to_text(self, test_data_dir):
        tokenizer = SentencePieceTokenizer(test_data_dir + self.model_name)

        text = "<cls> a b c <sep> e f g h i </s>"
        ids = tokenizer.text_to_ids(text)
        result = tokenizer.ids_to_text(ids)

        assert text == result

    @pytest.mark.unit
    def test_tokens_to_ids(self, test_data_dir):
        tokenizer = SentencePieceTokenizer(test_data_dir + self.model_name)

        tokens = ["<cls>", "a", "b", "c", "<sep>", "e", "f", "<sep>", "g", "h", "i", "</s>"]
        ids = tokenizer.tokens_to_ids(tokens)

        assert len(ids) == len(tokens)
        assert ids.count(tokenizer.token_to_id("<cls>")) == 1
        assert ids.count(tokenizer.token_to_id("</s>")) == 1
        assert ids.count(tokenizer.token_to_id("<sep>")) == 2

    @pytest.mark.unit
    def test_ids_to_tokens(self, test_data_dir):
        tokenizer = SentencePieceTokenizer(test_data_dir + self.model_name)

        tokens = ["<cls>", "a", "b", "c", "<sep>", "e", "f", "<sep>", "g", "h", "i", "</s>"]
        ids = tokenizer.tokens_to_ids(tokens)
        result = tokenizer.ids_to_tokens(ids)

        assert len(result) == len(tokens)

        for i in range(len(result)):
            assert result[i] == tokens[i]

    @pytest.mark.unit
    def test_encode(self, test_data_dir):
        tokenizer = SentencePieceTokenizer(test_data_dir + self.model_name)

        text = "This text should encode to sth more than `max_length` tokens..."
        result = tokenizer.encode(text)
        assert isinstance(result, list)

        max_length = 5
        result = tokenizer.encode(text, max_length=max_length)
        assert len(result) == max_length

        n = 2
        texts = [text for _ in range(n)]
        tokens_list = tokenizer.encode(texts, max_length=max_length)
        assert len(tokens_list) == n
        assert all(len(tokens) == max_length for tokens in tokens_list)

        result = tokenizer.encode(text, max_length=max_length, return_tensors="pt")
        assert isinstance(result, torch.LongTensor)
        assert result.size() == (1, max_length)

        with pytest.raises(AssertionError):
            tokenizer.encode(text, return_tensors="np")  # Only "pt" option implemented

    @pytest.mark.unit
    def test_decode(self, test_data_dir):
        tokenizer = SentencePieceTokenizer(test_data_dir + self.model_name)

        text = "ole ole [SEP] ole ola [SEP]"
        tokens = tokenizer.encode(text)
        assert text == tokenizer.decode(tokens)

        n = 8
        texts = [text for _ in range(n)]
        tokens_list = tokenizer.encode(texts)
        assert isinstance(tokens_list, list)
        assert len(tokens_list) == n
        for tokens in tokens_list:
            assert text == tokenizer.decode(tokens)

    @pytest.mark.unit
    def test_batch_encode_plus(self, test_data_dir):
        tokenizer = SentencePieceTokenizer(test_data_dir + self.model_name)

        texts = ["Welcome to NeMo!", "This is fun"]
        with pytest.raises(AssertionError):
            tokenizer.batch_encode_plus(texts[0])  # Input should be List[str]

        tokens_dict = tokenizer.batch_encode_plus(texts)
        assert isinstance(tokens_dict, dict)
        assert "input_ids" in tokens_dict
        assert tokens_dict["input_ids"] == tokenizer.encode(texts)

    @pytest.mark.unit
    def test_batch_decode(self, test_data_dir):
        tokenizer = SentencePieceTokenizer(test_data_dir + self.model_name)

        texts = ["Jaki to jest język?", "Kropka."]
        tokens = tokenizer.encode(texts)
        assert texts == tokenizer.batch_decode(tokens)
