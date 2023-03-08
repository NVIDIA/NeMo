# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import unittest

import pytest
from nemo_text_processing.text_normalization.normalize import Normalizer

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import IPATokenizer
from nemo.collections.tts.g2p.modules import IPAG2P
from nemo.collections.tts.inference.text_processors import BaseTextProcessor, IPATextTokenizer


class TestTextProcessors(unittest.TestCase):

    PHONEME_DICT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phoneme_dict", "test_dict.txt")

    @classmethod
    def setUpClass(cls):
        super(TestTextProcessors, cls).setUpClass()

        normalizer = Normalizer(lang="en", input_case="cased")
        cls.text_processor = BaseTextProcessor(normalizer)

        g2p = IPAG2P(
            TestTextProcessors.PHONEME_DICT_PATH, locale="en-US", apply_to_oov_word=lambda x: x, use_chars=True,
        )
        ipa_tokenizer = IPATokenizer(g2p=g2p)
        cls.ipa_text_tokenizer = IPATextTokenizer(tokenizer=ipa_tokenizer)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_preprocess_text(self):
        input_text = "Hello world."

        output_text = self.text_processor.preprocess_text(text=input_text)

        assert input_text == output_text

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_text(self):
        input_text = "Hello 1."
        expected_text = "Hello one."

        output_text = self.text_processor.normalize_text(text=input_text)

        assert expected_text == output_text

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_postprocess_text(self):
        input_text = "Hello world."

        output_text = self.text_processor.postprocess_text(text=input_text)

        assert input_text == output_text

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_text_tokenizer_convert_graphemes_to_phonemes(self):
        input_text = "Hello world."
        expected_text = [char for char in "həˈɫoʊ ˈwɝɫd."]

        output_text = self.ipa_text_tokenizer.convert_graphemes_to_phonemes(text=input_text)

        assert expected_text == output_text

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_text_tokenizer_convert_graphemes_to_phonemes_mixed(self):
        input_text = "Hello world."
        expected_text = [char for char in input_text.upper()]

        output_text = self.ipa_text_tokenizer.convert_graphemes_to_phonemes_mixed(text=input_text, phone_prob=0.0)

        assert expected_text == output_text

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_text_tokenizer_tokenize(self):
        input_text = "HELLO ˈwɝɫd."

        output_tokens = self.ipa_text_tokenizer.tokenize(text=input_text)
        output_text = self.ipa_text_tokenizer.tokenizer.decode(output_tokens).replace('|', '')

        assert input_text == output_text
        assert len(output_tokens) == len(input_text)
