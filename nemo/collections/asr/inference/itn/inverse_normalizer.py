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


import os
import re
from multiprocessing import Manager
from typing import List, Tuple

import diskcache
import pynini
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer, Normalizer
from nemo_text_processing.text_normalization.en.graph_utils import INPUT_CASED, INPUT_LOWER_CASED

from nemo.collections.asr.inference.utils.itn_utils import (
    DEFAULT_SEMIOTIC_CLASS,
    fallback_to_trivial_alignment,
    find_tokens,
    get_semiotic_class,
    split_text,
)
from nemo.utils import logging

IN_MEM_CACHE = Manager().dict(lock=False)


class AlignmentPreservingInverseNormalizer:
    LOWER_CASED = INPUT_LOWER_CASED
    UPPER_CASED = INPUT_CASED
    GRAMMAR = "itn"

    def __init__(
        self,
        input_case: str = LOWER_CASED,
        lang: str = "en",
        whitelist: str = None,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        max_number_of_permutations_per_split: int = 729,
    ):
        """
        Inverse normalizer that converts text from spoken to written form.
        Args:
            input_case: Input text capitalization, set to 'cased' if text contains capital letters.
                This flag affects normalization rules applied to the text. Note, `lower_cased` won't lower case input.
            lang: language specifying the ITN
            whitelist: path to a file with whitelist replacements. (each line of the file: written_form\tspoken_form\n),
                e.g. nemo_text_processing/inverse_text_normalization/en/data/whitelist.tsv
            cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
            overwrite_cache: set to True to overwrite .far files
            max_number_of_permutations_per_split: a maximum number
                of permutations which can be generated from input sequence of tokens.
        """
        self.itn_model = InverseNormalizer(
            lang=lang,
            input_case=input_case,
            whitelist=whitelist,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            max_number_of_permutations_per_split=max_number_of_permutations_per_split,
        )
        if cache_dir:
            self.DISK_TAG_CACHE = diskcache.Cache(os.path.join(cache_dir, "itn_tag_cache"))
            self.DISK_VERB_CACHE = diskcache.Cache(os.path.join(cache_dir, "itn_verb_cache"))
            self.caching_enabled = True
        else:
            self.DISK_TAG_CACHE = None
            self.DISK_VERB_CACHE = None
            self.caching_enabled = False

    def verbalize(self, tokens: List, sep: str) -> str | None:
        """
        Verbalizes tagged text.
        Args:
            tokens: (List) list of tokens
            sep: (str) word separator
        Returns:
            (str | None) Returns verbalized text. If verbalization fails, returns None.
        """
        split_tokens = self.itn_model._split_tokens_to_reduce_number_of_permutations(tokens)
        output_str = ""
        for s in split_tokens:
            try:
                tags_reordered = self.itn_model.generate_permutations(s)
                verbalizer_lattice = None
                for tagged_text_r in tags_reordered:
                    tagged_text_r = pynini.escape(tagged_text_r)

                    verbalizer_lattice = self.itn_model.find_verbalizer(tagged_text_r)
                    if verbalizer_lattice.num_states() != 0:
                        break

                if verbalizer_lattice is None:
                    return None

                verbalized_text = Normalizer.select_verbalizer(verbalizer_lattice)
                output_str += sep + verbalized_text
            except Exception as e:
                logging.warning("Failed to verbalize tagged text: " + str(e))
                return None

        output_str = output_str.strip(sep)
        return re.sub(r"({sep})+".format(sep=sep), sep, output_str)

    def tag(self, text: str, no_cache: bool = False) -> str:
        """
        Tags the input text.
        Args:
            text: (str) input text
            no_cache: (bool) whether to use cache
        Returns:
            (str) tagged text
        """
        if not no_cache and self.caching_enabled:
            # In-memory cache check
            if text in IN_MEM_CACHE:
                return IN_MEM_CACHE[text]

            # Disk cache check
            if text in self.DISK_TAG_CACHE:
                x = self.DISK_TAG_CACHE[text]
                IN_MEM_CACHE[text] = x
                return x

        text = text.strip()
        if not text:
            return text

        text = pynini.escape(text)
        tagged_lattice = self.itn_model.find_tags(text)
        tagged_text = Normalizer.select_tag(tagged_lattice)
        if self.caching_enabled:
            IN_MEM_CACHE[text] = tagged_text
            self.DISK_TAG_CACHE[text] = tagged_text
        return tagged_text

    def parse_and_verbalize(self, tagged_text: str, sep: str) -> Tuple[str, str]:
        """
        Tags and verbalizes the input text.
        Args:
            tagged_text: (str) tagged input text
            sep: (str) word separator
        Returns:
            (str, str) Returns the verbalized text, and the semiotic class.
        """
        if self.caching_enabled:
            # In-memory cache check
            if tagged_text in IN_MEM_CACHE:
                return IN_MEM_CACHE[tagged_text]

            # Disk cache check
            if tagged_text in self.DISK_VERB_CACHE:
                x = self.DISK_VERB_CACHE[tagged_text]
                IN_MEM_CACHE[tagged_text] = x
                return x

        self.itn_model.parser(tagged_text)
        tokens = self.itn_model.parser.parse()
        span_text = self.verbalize(tokens, sep)
        semiotic_class = DEFAULT_SEMIOTIC_CLASS if span_text is None else get_semiotic_class(tokens)
        if self.caching_enabled:
            self.DISK_VERB_CACHE[tagged_text] = (span_text, semiotic_class)
            IN_MEM_CACHE[tagged_text] = (span_text, semiotic_class)
        return span_text, semiotic_class

    def find_token_words(
        self, token: str, start_idx: int, input_words: List[str], sep: str
    ) -> Tuple[List[int], bool, int]:
        """
        Finds the words that make up the token.
        Args:
            token: (str) token
            start_idx: (int) start index
            input_words: (List[str]) list of input words
            sep: (str) word separator
        Returns:
            (Tuple) Returns a tuple of indices, success, and the new start index
        """
        indices, tmp_text, success = [], "", False
        length = len(input_words)
        for i in range(start_idx, length):
            tmp_text = tmp_text + sep + input_words[i] if tmp_text else input_words[i]
            tmp_tagged_text = self.tag(tmp_text)

            if tmp_tagged_text == token:
                indices.append(i)

                # Try to extend the token by one word
                if i + 1 < length:
                    extended_tmp_text = tmp_text + sep + input_words[i + 1]
                    extended_tmp_tagged_text = self.tag(extended_tmp_text)
                    if extended_tmp_tagged_text == token:
                        continue

                success = True
                break
            else:
                indices.append(i)

        return indices, success, i

    def find_alignment(
        self,
        tokens: List[str],
        input_words: List[str],
        sep: str,
        iwords: List[str],
        owords: List[str],
        word_alignment: List[Tuple],
    ) -> bool:
        """
        Finds the word alignment for the input text.
        Args:
            tokens: (List[str]) list of tokens
            input_words: (List[str]) list of input words
            sep: (str) word separator
            iwords: (List[str]) list of input words to be updated
            owords: (List[str]) list of output words to be updated
            word_alignment: (List[Tuple]) list of word alignments to be updated
        Returns:
            (bool) True if the word alignment is found, False otherwise
        """
        success = True
        token_start_idx = word_start_idx = 0
        iwords_len = owords_len = 0

        while token_start_idx < len(tokens):
            token = tokens[token_start_idx]
            current_word = input_words[word_start_idx]
            if token == f"tokens {{ name: \"{current_word}\" }}":
                word_alignment.append(([iwords_len], [owords_len], DEFAULT_SEMIOTIC_CLASS))
                iwords.append(current_word)
                owords.append(current_word)
                iwords_len += 1
                owords_len += 1
            else:
                indices, success, word_start_idx = self.find_token_words(token, word_start_idx, input_words, sep)
                if success:
                    span_text, semiotic_class = self.parse_and_verbalize(token, sep)
                    if span_text is None:
                        logging.warning(f"Failed to verbalize the token: {token}")
                        return False

                    span_words, n_span_words = split_text(span_text, sep)
                    word_alignment.append(
                        (
                            [iwords_len + i for i in range(len(indices))],
                            [owords_len + i for i in range(n_span_words)],
                            semiotic_class,
                        )
                    )
                    owords.extend(span_words)
                    iwords.extend([input_words[i] for i in indices])
                    iwords_len += len(indices)
                    owords_len += n_span_words
                else:
                    success = False
                    break

            token_start_idx += 1
            word_start_idx += 1

        return success

    def get_word_alignment(self, input: str | List[str], sep: str) -> Tuple[List[str], List[str], List[Tuple]]:
        """
        Returns a word alignment for the input text.
        Args:
            input: (str | List[str]) input text or list of input words
            sep: (str) word separator
        Returns:
            (Tuple) Returns a tuple of input words, output words, and a word alignment between input and output words
        """

        if isinstance(input, str):
            input_text = input
            input_words, n_words = split_text(input_text, sep)
        else:
            input_words, n_words = input, len(input)
            input_text = sep.join(input_words)

        # If input_text is empty, return empty lists
        if n_words == 0:
            return [], [], []

        # Tag the input text
        tagged_text = self.tag(input_text, no_cache=False)

        # Find the tokens in the tagged text
        tokens = find_tokens(tagged_text)

        # Find the word alignment
        iwords, owords, word_alignment = [], [], []
        success = self.find_alignment(
            tokens, input_words, sep, iwords=iwords, owords=owords, word_alignment=word_alignment
        )

        # If the word alignment is not found, fallback to the trivial alignment
        if not success:
            return fallback_to_trivial_alignment(input_words, i_shift=0, o_shift=0)

        return iwords, owords, word_alignment
