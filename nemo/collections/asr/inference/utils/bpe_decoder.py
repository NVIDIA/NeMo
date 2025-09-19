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


from dataclasses import dataclass
from typing import Callable, List, Set, Tuple

from nemo.collections.asr.inference.utils.constants import (
    POST_WORD_PUNCTUATION,
    ROUND_PRECISION,
    SENTENCEPIECE_UNDERSCORE,
)
from nemo.collections.asr.inference.utils.word import Word
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass(slots=True, frozen=True)
class Token:
    text: str
    timestep: float
    confidence: float
    idx: int
    is_start_of_word: bool


def refine_word_timestamp(
    word_parts: List[Token],
    next_word_parts: List[Token] | None,
    prev_word_end: float | None,
    punct_marks: Set,
    punct_marks_with_underscore: Set,
    word_boundary_tolerance: float,
) -> Tuple[float, float]:
    """
    Correct the start and end timestamps of the word parts
    Args:
        word_parts: (List[Token]) list of tokens
        next_word_parts: (List[Token]) list of tokens of the next word
        prev_word_end: (float) end timestamp of the previous word
        punct_marks: (Set) punctuation marks
        punct_marks_with_underscore: (Set) punctuation marks with underscore
        word_boundary_tolerance: (float) tolerance for word boundary
    Returns:
        (Tuple[int, int]) start and end timestamps of the word
    """
    start, end = word_parts[0].timestep, word_parts[-1].timestep

    # --- Correct the start timestamp if the first token is underscore or punctuation ---
    if word_parts[0].text in punct_marks_with_underscore:
        start = next((t.timestep for t in word_parts if t.text not in punct_marks_with_underscore), start)

    # --- Correct the end timestamp if the last token is punctuation ---
    if word_parts[-1].text in punct_marks:
        end = next((t.timestep for t in reversed(word_parts) if t.text not in punct_marks), end)

    # --- If the next word is close to the end of the current word, merge timestamps ---
    if next_word_parts and next_word_parts[0].is_start_of_word:
        if next_word_parts[0].timestep - end <= word_boundary_tolerance:
            end = next_word_parts[0].timestep

    delta = 0
    if prev_word_end is not None:
        if prev_word_end > start:
            delta = prev_word_end - start

    start = start + delta
    end = end + delta
    return start, end + (1 if start == end else 0)


class BPEDecoder:
    """
    BPEDecoder class for decoding BPE (Byte Pair Encoding) tokens into words with associated timestamps and confidence scores
    """

    def __init__(
        self,
        vocabulary: List[str],
        tokenizer: TokenizerSpec,
        confidence_aggregator: Callable,
        asr_supported_puncts: Set,
        word_boundary_tolerance: float,
        token_duration_in_secs: float,
    ):

        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.confidence_aggregator = confidence_aggregator
        self.asr_supported_puncts = asr_supported_puncts
        self.punct_marks_with_underscore = asr_supported_puncts.union({SENTENCEPIECE_UNDERSCORE})
        self.word_boundary_tolerance = word_boundary_tolerance
        self.token_duration_in_secs = token_duration_in_secs
        self.is_start_of_word_cache = {token: token.startswith(SENTENCEPIECE_UNDERSCORE) for token in self.vocabulary}

    def bpe_decode(self, tokens: List, timesteps: List, confidences: List) -> Tuple[List[Word], bool]:
        """
        Decodes BPE tokens into words with timestamps and confidence scores.
        Args:
            tokens (list): List of token indices.
            timesteps (list): List of token timesteps.
            confidences (list): List of token confidence scores.
        Returns:
            list: List of decoded words with text, start time, end time, and confidence score.
            merge_first_word: True if the first word should be merged with the last word stored in the state
        """

        if len(tokens) != len(timesteps) or len(tokens) != len(confidences):
            raise ValueError("tokens, timesteps and confidences must have the same length")

        if not tokens:
            return [], False

        parts = []
        word_parts = []
        for token_idx, token_ts, token_conf in zip(tokens, timesteps, confidences):

            token_text = self.vocabulary[token_idx]
            token = Token(
                text=token_text,
                timestep=token_ts,
                confidence=token_conf,
                idx=token_idx,
                is_start_of_word=self.is_start_of_word_cache[token_text],
            )

            # If a new word start is detected, push the previous word
            if token.is_start_of_word and len(word_parts) > 0:
                parts.append(word_parts)
                word_parts = []

            word_parts.append(token)

        # Append any leftover token_parts
        if len(word_parts) > 0:
            parts.append(word_parts)

        decoded_words = []
        # Keep track of the first word to merge with the last word stored in the state
        merge_first_word = not parts[0][0].is_start_of_word if len(parts) > 0 else False

        prev_word_end = None
        for i, word_parts in enumerate(parts):

            word_text = self.tokenizer.ids_to_text([token.idx for token in word_parts]).strip()

            # Ignore empty text
            if not word_text:
                continue

            # Append the post word punctuation to the previous word
            if word_text in POST_WORD_PUNCTUATION and len(decoded_words) > 0:
                prev_word = decoded_words[-1]
                prev_word.text += word_text
                continue

            # Refine timestamps
            word_start_tms, word_end_tms = refine_word_timestamp(
                word_parts,
                parts[i + 1] if (i + 1) < len(parts) else None,
                prev_word_end,
                self.asr_supported_puncts,
                self.punct_marks_with_underscore,
                self.word_boundary_tolerance,
            )
            prev_word_end = word_end_tms

            # Aggregate confidence
            word_conf = self.confidence_aggregator((token.confidence for token in word_parts))

            # Convert token timestamps to seconds
            start_sec = round(word_start_tms * self.token_duration_in_secs, ROUND_PRECISION)
            end_sec = round(word_end_tms * self.token_duration_in_secs, ROUND_PRECISION)

            decoded_words.append(Word(text=word_text, start=start_sec, end=end_sec, conf=word_conf))

        return decoded_words, merge_first_word
