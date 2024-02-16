# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from functools import cached_property
from pathlib import Path
from typing import Dict

from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model

__all__ = ['CanaryTokenizer']

# Default tokens for compatibility with Canary.
DEFAULT_TOKENS = ["<|nospeech|>", "<pad>", "<|endoftext|>", "<|startoftranscript|>", "<|pnc|>", "<|nopnc|>"]
DEFAULT_TASKS = [
    "transcribe",
    "translate",
]
DEFAULT_LANGUAGES = {
    "en": "english",
    "de": "german",
    "es": "spanish",
    "fr": "french",
}


class CanaryTokenizer(AggregateTokenizer):
    """
    Thin wrapper around AggregateTokenizer to provide quick access to special tokens
    """

    def __init__(self, tokenizers: Dict, tasks=DEFAULT_TASKS, langs=DEFAULT_LANGUAGES):
        # Formats tokens
        langs, tasks = [f"<|{l}|>" for l in langs], [f"<|{t}|>" for t in tasks]
        langs.sort(), tasks.sort()
        all_tokens = DEFAULT_TOKENS + tasks + langs

        # Creates spl_tokens if not passed
        if 'spl_tokens' not in tokenizers:
            tokenizers['spl_tokens'] = CanaryTokenizer.build_special_tokenizer(all_tokens)
        super().__init__(tokenizers)

        # for easy access of special tokens
        self.special_tokens = {}
        for special in all_tokens:
            self.special_tokens[special] = self.token_to_id(special, lang_id='spl_tokens')

    @cached_property
    def eos_id(self) -> int:
        return self.special_tokens["<|endoftext|>"]

    @cached_property
    def bos_id(self) -> int:
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def nospeech_id(self) -> int:
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def pad_id(self) -> int:
        return self.special_tokens["<pad>"]

    def to_language_id(self, language):
        if token_id := self.special_tokens.get(f"<|{language}|>", None):
            return token_id
        raise KeyError(f"Language {language} not found in tokenizer.")

    def to_task_id(self, task):
        if token_id := self.special_tokens.get(f"<|{task}|>", None):
            return token_id
        raise KeyError(f"Task {task} not found in tokenizer.")

    @staticmethod
    def build_special_tokenizer(tokens, output_dir: str | Path = None) -> SentencePieceTokenizer:
        is_temp = False
        if output_dir is None:
            is_temp = True
            output_dir = "__tmp_canary_spl_tokenizer__"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=False if is_temp else True, parents=True)
        text_path = output_dir / "train_text.txt"
        train_text = "\n".join(tokens)
        text_path.write_text(train_text)
        model_path = output_dir / "tokenizer.model"
        create_spt_model(
            str(text_path),
            vocab_size=len(tokens) + 2,
            sample_size=-1,
            do_lower_case=False,
            output_dir=str(output_dir),
            user_defined_symbols=tokens,
        )
        spl_tokenizer = SentencePieceTokenizer(str(model_path))
        if is_temp:
            os.remove(os.path.join(output_dir, "tokenizer.vocab"))
            os.remove(os.path.join(output_dir, "vocab.txt"))
            os.remove(os.path.join(output_dir, "tokenizer.model"))
            os.remove(os.path.join(output_dir, "train_text.txt"))
            Path.rmdir(output_dir)
        return spl_tokenizer
