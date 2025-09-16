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


from typing import List, Set, Union

from nemo.collections.common.inference.utils.constants import (
    BIG_EPSILON,
    DEFAULT_SEMIOTIC_CLASS,
    SEP_REPLACEABLE_PUNCTUATION,
)


class Word:
    __slots__ = ['_text', '_start', '_end', '_conf', '_semiotic_class', "_channel_id"]

    def __init__(
        self, text: str, start: float, end: float, conf: float, semiotic_class: str = DEFAULT_SEMIOTIC_CLASS
    ) -> None:
        """
        Initialize a Word instance.

        Args:
            text: The text content of the word
            start: Start time in seconds
            end: End time in seconds
            conf: Confidence score (typically 0.0 to 1.0)
            semiotic_class: Semantic classification of the word
            channel_id: Audio channel identifier

        Raises:
            ValueError: If start >= end or if confidence is negative
            TypeError: If text is not a string
        """
        self._validate_init_params(text, start, end, conf, semiotic_class, strict=False)

        self._text = text
        self._start = start
        self._end = end
        self._conf = conf
        self._semiotic_class = semiotic_class

        # Channel id is set to "A" by default
        # It is used to identify the channel of the word in the CTM file
        self._channel_id = "A"

    @staticmethod
    def _validate_init_params(
        text: str, start: float, end: float, conf: float, semiotic_class: str, strict: bool = False
    ) -> None:
        """Validate initialization parameters."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if not isinstance(start, (int, float)):
            raise TypeError(f"start must be numeric, got {type(start).__name__}")
        if not isinstance(end, (int, float)):
            raise TypeError(f"end must be numeric, got {type(end).__name__}")
        if not isinstance(conf, (int, float)):
            raise TypeError(f"conf must be numeric, got {type(conf).__name__}")
        if not isinstance(semiotic_class, str):
            raise TypeError(f"semiotic_class must be a string, got {type(semiotic_class).__name__}")

        if strict:
            if start >= end:
                raise ValueError(f"start time ({start}) must be less than end time ({end})")
            if conf < 0 or conf > 1:
                raise ValueError(f"confidence ({conf}) must be between 0 and 1")

    @property
    def text(self) -> str:
        """Text content of the word."""
        return self._text

    @property
    def channel_id(self) -> str:
        """Channel id of the word."""
        return self._channel_id

    @property
    def start(self) -> float:
        """Start time of the word in seconds."""
        return self._start

    @property
    def end(self) -> float:
        """End time of the word in seconds."""
        return self._end

    @property
    def duration(self) -> float:
        """Duration of the word in seconds."""
        return self._end - self._start

    @property
    def conf(self) -> float:
        """Confidence score of the word."""
        return self._conf

    @property
    def semiotic_class(self) -> str:
        """Semiotic class of the word."""
        return self._semiotic_class

    @text.setter
    def text(self, value: str) -> None:
        """Set the text content of the word."""
        if not isinstance(value, str):
            raise TypeError(f"text must be a string, got {type(value).__name__}")
        self._text = value

    @start.setter
    def start(self, value: float) -> None:
        """Set the start time."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"start time must be numeric, got {type(value).__name__}")
        self._start = value

    @end.setter
    def end(self, value: float) -> None:
        """Set the end time."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"end must be numeric, got {type(value).__name__}")
        self._end = value

    @conf.setter
    def conf(self, value: float) -> None:
        """Set the confidence score."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"conf must be numeric, got {type(value).__name__}")
        if value < 0 or value > 1:
            raise ValueError(f"confidence ({value}) must be between 0 and 1")
        self._conf = value

    @semiotic_class.setter
    def semiotic_class(self, value: str) -> None:
        """Set the semiotic classification."""
        if not isinstance(value, str):
            raise TypeError(f"semiotic_class must be a string, got {type(value).__name__}")
        self._semiotic_class = value

    def copy(self) -> 'Word':
        """
        Create a deep copy of this Word instance.

        Returns:
            A new Word instance with identical properties
        """
        return Word(text=self.text, start=self.start, end=self.end, conf=self.conf, semiotic_class=self.semiotic_class)

    def capitalize(self) -> None:
        """Capitalize the text of the word."""
        self._text = self._text.capitalize()

    def with_normalized_text(self, punct_marks: Set[str], sep: str = "") -> 'Word':
        """
        Create a new Word with normalized text (punctuation removed/replaced and lowercased).

        Args:
            punct_marks: Set of punctuation marks to process
            sep: Separator to replace certain punctuation marks

        Returns:
            New Word instance with normalized text
        """
        replace_map = {mark: sep if mark in SEP_REPLACEABLE_PUNCTUATION else "" for mark in punct_marks}
        trans_table = str.maketrans(replace_map)
        normalized_text = self.text.translate(trans_table).lower()

        # Return new instance instead of modifying in place
        word_copy = self.copy()
        word_copy.text = normalized_text
        return word_copy

    def normalize_text_inplace(self, punct_marks: Set[str], sep: str = "") -> None:
        """
        Normalize text in place (punctuation removed/replaced and lowercased).

        Args:
            punct_marks: Set of punctuation marks to process
            sep: Separator to replace certain punctuation marks

        Note:
            This method modifies the current instance. Consider using
            with_normalized_text() for a functional approach.
        """
        replace_map = {mark: sep if mark in SEP_REPLACEABLE_PUNCTUATION else "" for mark in punct_marks}
        trans_table = str.maketrans(replace_map)
        self.text = self.text.translate(trans_table).lower()

    def __eq__(self, other: object) -> bool:
        """
        Check equality with another Word instance.

        Args:
            other: Another object to compare with

        Returns:
            True if both instances represent the same word
        """
        if not isinstance(other, Word):
            raise NotImplementedError(f"Cannot compare Word with {type(other)}")

        return (
            self.text == other.text
            and self.channel_id == other.channel_id
            and self.semiotic_class == other.semiotic_class
            and abs(self.start - other.start) < BIG_EPSILON
            and abs(self.end - other.end) < BIG_EPSILON
            and abs(self.conf - other.conf) < BIG_EPSILON
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"'{self.text}' [{self.start:.2f}-{self.end:.2f}s] (conf: {self.conf:.2f})"

    def get_ctm_line(self) -> str:
        """
        Generate a CTM (Conversation Time Marked) format line.

        Returns:
            CTM formatted string: "channel_id start duration text confidence semiotic_class"
        """
        return (
            f"{self.channel_id} {round(self.start, 2)} {round(self.duration, 2)} "
            f"{self.text} {self.conf} {self.semiotic_class}"
        )


def join_words(words: List[List[Word]], sep: str) -> List[str]:
    """
    Join the words to form transcriptions.

    Args:
        words: List of word sequences to join
        sep: Separator to use when joining words

    Returns:
        List of transcriptions, one for each word sequence
    """
    return [sep.join([w.text for w in items]) for items in words]


def normalize_words_inplace(words: Union[List[Word], List[List[Word]]], punct_marks: Set[str], sep: str = ' ') -> None:
    """
    Normalize text in words by removing punctuation and converting to lowercase.

    This function modifies the words in-place by calling normalize_text_inplace
    on each Word object. It handles both flat lists of words and nested lists.

    Args:
        words: List of Word objects or list of lists of Word objects
        punct_marks: Set of punctuation marks to be processed
        sep: Separator to replace certain punctuation marks (default: ' ')

    Note:
        This function modifies the input words in-place. The original text
        content of the words will be permanently changed.
    """
    for item in words:
        if isinstance(item, list):
            for word in item:
                word.normalize_text_inplace(punct_marks, sep)
        elif isinstance(item, Word):
            item.normalize_text_inplace(punct_marks, sep)
        else:
            raise ValueError(f"Invalid item type: {type(item)}. Expected `Word` or `List[Word]`.")
