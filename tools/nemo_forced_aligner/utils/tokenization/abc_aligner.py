from abc import ABC, abstractmethod
from typing import Type, Union

from utils.units import Alignment, Segment, Token, Word

from nemo.utils import logging


class BaseAligner(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def text_to_ids(self, text: str):
        pass

    @abstractmethod
    def text_to_tokens(self, text: str):
        pass

    @staticmethod
    def text_to_segments(text, separator: str = None):
        if separator:
            segmented_text = text.split(separator)
        else:
            segmented_text = [text]

        segmented_text = [seg for seg in [seg.strip() for seg in segmented_text] if len(seg) > 0]

        return segmented_text

    def is_alignable(self, text, T: int):
        if text is None or len(text) == 0:
            logging.info(f"utterance is empty - we will not generate" " any output alignment files for this utterance")
            return False

        tokens = self.text_to_ids(text)
        n_token_repetitions = 0
        for i_tok in range(1, len(tokens)):
            if tokens[i_tok] == tokens[i_tok - 1]:
                n_token_repetitions += 1

        if len(tokens) + n_token_repetitions > T:
            logging.info(
                f"Utterance has too many tokens compared to the audio file duration."
                " Will not generate output alignment files for this utterance."
            )
            return False
        return True

    def align_unit(self, unit_text: str, unit_type: Type[Union[Segment, Word, Token]], unit_s_pointer: int = 0):
        if unit_type == Segment or unit_type == Word:
            unit_tokens = self.text_to_tokens(unit_text)
            unit = unit_type(text=unit_text, s_start=unit_s_pointer, s_end=unit_s_pointer + len(unit_tokens) * 2 - 2)

            unit_s_pointer += len(unit_tokens) * 2
        else:
            unit_tokens = [unit_text]
            unit = unit_type(text=unit_text, s_start=unit_s_pointer, s_end=unit_s_pointer)
            unit_s_pointer += 1

        return unit, unit_tokens, unit_s_pointer

    @abstractmethod
    def align(self, alignment_obj: Alignment, **kwargs):
        pass
