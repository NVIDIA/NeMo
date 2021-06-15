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

from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer


class EnJaProcessor:
    """
    Tokenizer, Detokenizer and Normalizer utilities for Japanese & English
    Args:
        lang_id: One of ['en', 'ja'].
    """

    def __init__(self, lang_id: str):
        self.lang_id = lang_id

    def detokenize(self, tokens: List[str]) -> str:
        return ' '.join(tokens)

    def tokenize(self, text) -> str:
        return text

    def normalize(self, text) -> str:
        return text
