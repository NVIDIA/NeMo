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

from sacremoses import MosesDetokenizer, MosesTokenizer

from nemo.collections.common.tokenizers.sentencepiece_detokenizer import SentencePieceDetokenizer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer


class EnJaDetokenizer:
    """
    Deokenizer for Japanese & English that undoes `EnJaTokenizer` tokenization.
    Args:
        lang_id: One of ['en', 'ja'].
    """

    def __init__(self, lang_id: str):
        self.moses_detokenizer = MosesDetokenizer(lang=lang_id)
        self.sp_detokenizer = SentencePieceDetokenizer()

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


class EnJaTokenizer:
    """
    Tokenizer for Japanese & English that does Moses tokenization followed by SentencePiece
    Args:
        sp_tokenizer_model_path: String path to a sentencepiece model
        lang_id: One of ['en', 'ja'].
    """

    def __init__(self, sp_tokenizer_model_path: str, lang_id: str):
        self.moses_tokenizer = MosesTokenizer(lang=lang_id)
        self.sp_tokenizer = SentencePieceTokenizer(model_path=sp_tokenizer_model_path)

    def sp_tokenize(self, text: str) -> str:
        return ' '.join(self.sp_tokenizer.text_to_tokens(text))

    def tokenize(self, text, escape=False, return_str=False):
        """
        Tokenizes text using Moses -> Sentencepiece.
        """
        text = self.moses_tokenizer.tokenize(text, escape=escape, return_str=True)
        text = self.sp_tokenize(text)
        return text if return_str else text.split()
