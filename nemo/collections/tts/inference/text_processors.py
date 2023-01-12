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

from typing import List

from nemo.collections.tts.inference.pipeline import TextProcessor, TextTokenizer, GraphemeToPhoneme
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import IPATokenizer
from nemo_text_processing.text_normalization.normalize import Normalizer


class BaseTextProcessor(TextProcessor):
    def __init__(self, normalizer: Normalizer):
        self.normalizer = normalizer

    def preprocess_text(self, text: str) -> str:
        return text

    def normalize_text(self, text: str) -> str:
        normalized_text = self.normalizer.normalize(text, punct_pre_process=True, punct_post_process=True)
        return normalized_text

    def postprocess_text(self, text: str) -> str:
        return text


class IPATextTokenizer(TextTokenizer, GraphemeToPhoneme):
    def __init__(self, tokenizer: IPATokenizer):
        self.tokenizer = tokenizer

    def convert_graphemes_to_phonemes(self, text: str) -> str:
        if not hasattr(self.tokenizer, "set_phone_prob"):
            phonemes = self.tokenizer.g2p(text)
        else:
            with self.tokenizer.set_phone_prob(prob=1.0):
                phonemes = self.tokenizer.g2p(text)
        return phonemes

    def convert_graphemes_to_phonemes_mixed(self, text: str, phone_prob: float = 0.5) -> str:
        with self.tokenizer.set_phone_prob(prob=phone_prob):
            text = self.tokenizer.g2p(text)
        return text

    def tokenize(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode_from_g2p(text)
        return tokens
