# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from sacremoses import MosesDetokenizer, MosesTokenizer, MosesPunctNormalizer

from nemo.collections.common.tokenizers.sentencepiece_detokenizer import SentencePieceDetokenizer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer


class EnJaProcessor:
    """
    Tokenizer, Detokenizer and Normalizer utilities for Japanese & English
    Args:
        sp_tokenizer_model_path: Path to sentencepiece tokenizer model file.
        lang_id: One of ['en', 'ja'].
    """
    def __init__(self, sp_tokenizer_model_path:str, lang_id: str):
        self.moses_tokenizer = MosesTokenizer(lang=lang_id)
        self.moses_detokenizer = MosesDetokenizer(lang=lang_id)
        self.sp_detokenizer = SentencePieceDetokenizer()
        self.sp_tokenizer = SentencePieceTokenizer(model_path=sp_tokenizer_model_path) \
            if sp_tokenizer_model_path is not None else None
        self.normalizer = MosesPunctNormalizer(
            lang=lang_id, pre_replace_unicode_punct=True, post_remove_control_chars=True
        )

    def detokenize(self, tokens: List[str]) -> str:
        """
        Detokenizes a list of tokens
        Args:
            tokens: list of strings as tokens
        Returns:
            detokenized Japanese or English string
        """
        text = self.sp_detokenizer.detokenize(tokens)
        return self.moses_detokenizer.detokenize(text)

    def sp_tokenize(self, text: str) -> str:
        return ' '.join(self.sp_tokenizer.text_to_tokens(text))

    def tokenize(self, text):
        """
        Tokenizes text using Moses -> Sentencepiece.
        """
        if self.sp_tokenizer is None:
            raise ValueError("Need valid sp_tokenizer_model_path, found None")
        text = self.moses_tokenizer.tokenize(text, escape=False, return_str=True)
        return self.sp_tokenize(text)
    
    def normalize(self, text):
        return self.normalizer.normalize(text)
