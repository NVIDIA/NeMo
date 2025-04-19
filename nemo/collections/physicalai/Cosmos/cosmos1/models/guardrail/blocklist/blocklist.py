# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
import string
from difflib import SequenceMatcher

import nltk
from better_profanity import profanity

from cosmos1.models.guardrail.blocklist.utils import read_keyword_list_from_dir, to_ascii
from cosmos1.models.guardrail.common.core import ContentSafetyGuardrail, GuardrailRunner
from cosmos1.utils import log, misc

DEFAULT_CHECKPOINT_DIR = "checkpoints/Cosmos-1.0-Guardrail/blocklist"
CENSOR = misc.Color.red("*")


class Blocklist(ContentSafetyGuardrail):
    def __init__(
        self,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        guardrail_partial_match_min_chars: int = 4,
        guardrail_partial_match_letter_count: float = 0.5,
    ) -> None:
        nltk.data.path.append(os.path.join(checkpoint_dir, "nltk_data"))
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.profanity = profanity
        self.checkpoint_dir = checkpoint_dir
        self.guardrail_partial_match_min_chars = guardrail_partial_match_min_chars
        self.guardrail_partial_match_letter_count = guardrail_partial_match_letter_count

        # Load blocklist and whitelist keywords
        self.blocklist_words = read_keyword_list_from_dir(os.path.join(self.checkpoint_dir, "custom"))
        self.whitelist_words = read_keyword_list_from_dir(os.path.join(self.checkpoint_dir, "whitelist"))
        self.exact_match_words = read_keyword_list_from_dir(os.path.join(self.checkpoint_dir, "exact_match"))

        self.profanity.load_censor_words(custom_words=self.blocklist_words, whitelist_words=self.whitelist_words)
        log.debug(f"Loaded {len(self.blocklist_words)} words/phrases from blocklist")
        log.debug(f"Whitelisted {len(self.whitelist_words)} words/phrases from whitelist")
        log.debug(f"Loaded {len(self.exact_match_words)} exact match words/phrases from blocklist")

    def uncensor_whitelist(self, input_prompt: str, censored_prompt: str) -> str:
        """Explicitly uncensor words that are in the whitelist."""
        input_words = input_prompt.split()
        censored_words = censored_prompt.split()
        whitelist_words = set(self.whitelist_words)
        for i, token in enumerate(input_words):
            if token.strip(string.punctuation).lower() in whitelist_words:
                censored_words[i] = token
        censored_prompt = " ".join(censored_words)
        return censored_prompt

    def censor_prompt(self, input_prompt: str) -> tuple[bool, str]:
        """Censor the prompt using the blocklist with better-profanity fuzzy matching.

        Args:
            input_prompt: input prompt to censor

        Returns:
            bool: True if the prompt is blocked, False otherwise
            str: A message indicating why the prompt was blocked
        """
        censored_prompt = self.profanity.censor(input_prompt, censor_char=CENSOR)
        # Uncensor whitelisted words that were censored from blocklist fuzzy matching
        censored_prompt = self.uncensor_whitelist(input_prompt, censored_prompt)
        if CENSOR in censored_prompt:
            return True, f"Prompt blocked by censorship: Censored Prompt: {censored_prompt}"
        return False, ""

    @staticmethod
    def check_partial_match(
        normalized_prompt: str, normalized_word: str, guardrail_partial_match_letter_count: float
    ) -> tuple[bool, str]:
        """
        Check robustly if normalized word and the matching target have a difference of up to guardrail_partial_match_letter_count characters.

        Args:
            normalized_prompt: a string with many words
            normalized_word: a string with one or multiple words, its length is smaller than normalized_prompt
            guardrail_partial_match_letter_count: maximum allowed difference in characters (float to allow partial characters)

        Returns:
            bool: True if a match is found, False otherwise
            str: A message indicating why the prompt was blocked
        """
        prompt_words = normalized_prompt.split()
        word_length = len(normalized_word.split())
        max_similarity_ratio = (len(normalized_word) - float(guardrail_partial_match_letter_count)) / float(
            len(normalized_word)
        )

        for i in range(len(prompt_words) - word_length + 1):
            # Extract a substring from the prompt with the same number of words as the normalized_word
            substring = " ".join(prompt_words[i : i + word_length])
            similarity_ratio = SequenceMatcher(None, substring, normalized_word).ratio()
            if similarity_ratio >= max_similarity_ratio:
                return (
                    True,
                    f"Prompt blocked by partial match blocklist: Prompt: {normalized_prompt}, Partial Match Word: {normalized_word}",
                )

        return False, ""

    @staticmethod
    def check_against_whole_word_blocklist(
        prompt: str,
        blocklist: list[str],
        guardrail_partial_match_min_chars: int = 4,
        guardrail_partial_match_letter_count: float = 0.5,
    ) -> bool:
        """
        Check if the prompt contains any whole words from the blocklist.
        The match is case insensitive and robust to multiple spaces between words.

        Args:
            prompt: input prompt to check
            blocklist: list of words to check against
            guardrail_partial_match_min_chars: minimum number of characters in a word to check for partial match
            guardrail_partial_match_letter_count: maximum allowed difference in characters for partial match

        Returns:
            bool: True if a match is found, False otherwise
            str: A message indicating why the prompt was blocked
        """
        # Normalize spaces and convert to lowercase
        normalized_prompt = re.sub(r"\s+", " ", prompt).strip().lower()

        for word in blocklist:
            # Normalize spaces and convert to lowercase for each blocklist word
            normalized_word = re.sub(r"\s+", " ", word).strip().lower()

            # Use word boundaries to ensure whole word match
            if re.search(r"\b" + re.escape(normalized_word) + r"\b", normalized_prompt):
                return True, f"Prompt blocked by exact match blocklist: Prompt: {prompt}, Exact Match Word: {word}"

            # Check for partial match if the word is long enough
            if len(normalized_word) >= guardrail_partial_match_min_chars:
                match, message = Blocklist.check_partial_match(
                    normalized_prompt, normalized_word, guardrail_partial_match_letter_count
                )
                if match:
                    return True, message

        return False, ""

    def is_safe(self, input_prompt: str = "") -> tuple[bool, str]:
        """Check if the input prompt is safe using the blocklist."""
        # Check if the input is empty
        if not input_prompt:
            return False, "Input is empty"
        input_prompt = to_ascii(input_prompt)

        # Check full sentence for censored words
        censored, message = self.censor_prompt(input_prompt)
        if censored:
            return False, message

        # Check lemmatized words for censored words
        tokens = nltk.word_tokenize(input_prompt)
        lemmas = [self.lemmatizer.lemmatize(token) for token in tokens]
        lemmatized_prompt = " ".join(lemmas)
        censored, message = self.censor_prompt(lemmatized_prompt)
        if censored:
            return False, message

        # Check for exact match blocklist words
        censored, message = self.check_against_whole_word_blocklist(
            input_prompt,
            self.exact_match_words,
            self.guardrail_partial_match_min_chars,
            self.guardrail_partial_match_letter_count,
        )
        if censored:
            return False, message

        # If all these checks pass, the input is safe
        return True, "Input is safe"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Path to the Blocklist checkpoint folder",
        default=DEFAULT_CHECKPOINT_DIR,
    )
    return parser.parse_args()


def main(args):
    blocklist = Blocklist(checkpoint_dir=args.checkpoint_dir)
    runner = GuardrailRunner(safety_models=[blocklist])
    with misc.timer("blocklist safety check"):
        safety, message = runner.run_safety_check(args.prompt)
    log.info(f"Input is: {'SAFE' if safety else 'UNSAFE'}")
    log.info(f"Message: {message}") if not safety else None


if __name__ == "__main__":
    args = parse_args()
    main(args)
