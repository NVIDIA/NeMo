from typing import List

from utils.tokenization.abc import BaseAligner
from utils.units import Alignment, Segment, Word, Token, BlankToken
from utils import constants


class TokenBasedAligner(BaseAligner):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer

    def text_to_ids(self, text: str):
        return self.tokenizer.text_to_ids(text)

    def text_to_tokens(self, text: str):
        return self.tokenizer.text_to_tokens(text)

    @staticmethod
    def _is_sub_or_superscript_pair(ref_text, text):
        """returns True if ref_text is a subscript or superscript version of text"""
        sub_or_superscript_to_num = {
            "⁰": "0",
            "¹": "1",
            "²": "2",
            "³": "3",
            "⁴": "4",
            "⁵": "5",
            "⁶": "6",
            "⁷": "7",
            "⁸": "8",
            "⁹": "9",
            "₀": "0",
            "₁": "1",
            "₂": "2",
            "₃": "3",
            "₄": "4",
            "₅": "5",
            "₆": "6",
            "₇": "7",
            "₈": "8",
            "₉": "9",
        }

        if text in sub_or_superscript_to_num:
            if sub_or_superscript_to_num[text] == ref_text:
                return True
        return False

    @staticmethod
    def _restore_token_case(word: str, word_tokens: List[str]):
        # remove repeated "▁" and "_" from word as that is what the tokenizer will do
        while "▁▁" in word:
            word = word.replace("▁▁", "▁")

        while "__" in word:
            word = word.replace("__", "_")

        word_tokens_cased = []
        word_char_pointer = 0

        for token in word_tokens:
            token_cased = ""

            for token_char in token:
                if token_char == word[word_char_pointer]:
                    token_cased += token_char
                    word_char_pointer += 1

                else:
                    if token_char.upper() == word[word_char_pointer] or TokenBasedAligner._is_sub_or_superscript_pair(
                        token_char, word[word_char_pointer]
                    ):
                        token_cased += token_char.upper()
                        word_char_pointer += 1
                    else:
                        if token_char == "▁" or token_char == "_":
                            if word[word_char_pointer] == "▁" or word[word_char_pointer] == "_":
                                token_cased += token_char
                                word_char_pointer += 1
                            elif word_char_pointer == 0:
                                token_cased += token_char

                        else:
                            raise RuntimeError(
                                f"Unexpected error - failed to recover capitalization of tokens for word {word}"
                            )

            word_tokens_cased.append(token_cased)

        return word_tokens_cased

    def align(self, alignment: Alignment, T: int, separator: str = None):
        if not self.is_alignable(alignment.text, T):
            return

        alignment.segments_and_tokens.append(BlankToken(s_start=0, s_end=0))
        segment_s_pointer = 1
        word_s_pointer = 1

        alignment.token_ids_with_blanks = [constants.BLANK_ID]
        token_s_pointer = 1

        text_segments = self.text_to_segments(alignment.text, separator)

        for text_segment in text_segments:
            segment, segment_tokens, segment_s_pointer = self.align_unit(
                unit_text=text_segment, unit_type=Segment, unit_s_pointer=segment_s_pointer
            )

            segment_words = segment.text.split()

            for word_i, _word in enumerate(segment_words):
                word, word_tokens, word_s_pointer = self.align_unit(unit_text=_word,
                                                                    unit_type=Word,
                                                                    unit_s_pointer=word_s_pointer)
                
                word_tokens_cased = self._restore_token_case(_word, word_tokens)
                word_tokens_ids = self.text_to_ids(_word)

                for token_i, (_token, token_cased, token_id) in enumerate(zip(word_tokens, word_tokens_cased, word_tokens_ids)):
                    alignment.token_ids_with_blanks.extend([token_id, constants.BLANK_ID])
                    
                    token, _, token_s_pointer = self.align_unit(unit_text=_token,
                                                                unit_type=Token,
                                                                unit_s_pointer=token_s_pointer)
                    token.text_cased = token_cased

                    word.tokens.append(token)
                    if token_i < len(word_tokens) - 1:
                        token_s_pointer += 1
                        word.tokens.append(BlankToken(s_start=token_s_pointer, s_end=token_s_pointer))

                segment.words_and_tokens.append(word)
                if word_i < len(segment_words) - 1:
                    segment.words_and_tokens.append(BlankToken(s_start=token_s_pointer, s_end=token_s_pointer))

            alignment.segments_and_tokens.extend([segment, BlankToken(s_start=token_s_pointer, s_end=token_s_pointer)])

        return
