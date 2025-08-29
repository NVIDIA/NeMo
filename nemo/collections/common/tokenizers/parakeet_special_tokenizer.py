# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import unicodedata

from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

__all__ = ["ParakeetBPETokenizer"]


class ParakeetBPETokenizer(SentencePieceTokenizer):
    """
    SentencePiece tokenizer wrapper for Parakeet-style ASR with event tags.

    It overrides supported_punctuation used by RNNT decoding to avoid stripping the
    space before '[' and ']' so bracketed event tokens like "[laugh]" can be
    rendered with a preceding space (e.g., "hello [laugh]") if desired.
    """

    @property
    def supported_punctuation(self):
        # All unicode punctuation characters present in the vocabulary except square brackets
        all_punct = {char for token in self.vocab for char in token if unicodedata.category(char).startswith("P")}
        return all_punct.difference({"[", "]"})



