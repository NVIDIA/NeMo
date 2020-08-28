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

from nemo.collections.common.tokenizers.char_tokenizer import CharTokenizer

__all__ = ['WordTokenizer']


class WordTokenizer(CharTokenizer):
    def __init__(
        self,
        vocab_file: str,
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
    ):
        """
        Args:
            vocab_file: path to file with vocabulary which consists
                of characters separated by \n
            bos_token: the beginning of sequence token
            eos_token: the end of sequence token
            pad_token: token to use for padding
            unk_token: token to use for unknown tokens
        """

        super().__init__(vocab_file, bos_token, eos_token, pad_token, unk_token)

    def text_to_tokens(self, text):
        token_candidates = text.strip().split()
        tokens = []
        for token in token_candidates:
            if token in self.vocab:
                tokens.append(token)
            else:
                tokens.append(self.unk_token)
        return tokens

    def ids_to_text(self, ids):
        ids_ = [id_ for id_ in ids if id_ not in self.special_tokens]
        return " ".join(self.ids_to_tokens(ids_))
