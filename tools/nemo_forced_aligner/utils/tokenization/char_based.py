from utils import constants
from utils.tokenization.abc import BaseAligner
from utils.units import Alignment, BlankToken, Segment, SpaceToken, Token, Word


class CharBasedAligner(BaseAligner):
    def __init__(self, vocabulary, **kwargs):
        super().__init__(**kwargs)
        self.vocabulary = vocabulary

    def text_to_ids(self, text: str):
        tokens = []
        for character in text:
            if character in self.vocabulary:
                tokens.append(self.vocabulary.index(character))
            else:
                tokens.append(len(self.vocabulary))

        return tokens

    def text_to_tokens(self, text: str):
        return list(text)

    def align(self, alignment: Alignment, T: int, separator: str = None):
        if not self.is_alignable(alignment.text):
            return

        alignment.segments_and_tokens.append(BlankToken(s_start=0, s_end=0))
        segment_s_pointer = 1
        word_s_pointer = 1

        alignment.token_ids_with_blanks = [constants.BLANK_ID]
        token_s_pointer = 1

        text_segments = self.text_to_segments(alignment.text, separator)

        for i_segment, text_segment in enumerate(text_segments):
            segment, segment_tokens, segment_s_pointer = self.align_unit(
                unit_text=text_segment, unit_type=Segment, unit_s_pointer=segment_s_pointer
            )

            segment_words = segment.text.split()

            for word_i, _word in enumerate(segment_words):
                word, word_tokens, word_s_pointer = self.align_unit(
                    unit=_word, unit_type=Word, unit_s_pointer=word_s_pointer
                )

                word_tokens_ids = self.text_to_ids(_word)

                for token_i, (_token, token_id) in enumerate(zip(word_tokens, word_tokens_ids)):
                    alignment.token_ids_with_blanks.extend([token_id])

                    token, _, token_s_pointer = self.align_unit(
                        unit=_token, unit_type=Token, unit_s_pointer=token_s_pointer
                    )
                    token.text_cased = _token

                    word.tokens.append(token)
                    if token_i < len(word_tokens) - 1:
                        token_s_pointer += 1
                        alignment.token_ids_with_blanks.extend([constants.BLANK_ID])
                        word.tokens.append(BlankToken(s_start=token_s_pointer, s_end=token_s_pointer))

                segment.words_and_tokens.append(word)
                if word_i < len(segment_words) - 1:

                    token_s_pointer += 1
                    alignment.token_ids_with_blanks.extend([constants.BLANK_ID])
                    segment.words_and_tokens.append(BlankToken(s_start=token_s_pointer, s_end=token_s_pointer))

                    token_s_pointer += 1
                    alignment.token_ids_with_blanks.extend([constants.SPACE_ID])
                    segment.words_and_tokens.append(SpaceToken(s_start=token_s_pointer, s_end=token_s_pointer))

                    token_s_pointer += 1
                    alignment.token_ids_with_blanks.extend([constants.BLANK_ID])
                    segment.words_and_tokens.append(BlankToken(s_start=token_s_pointer, s_end=token_s_pointer))

            alignment.segments_and_tokens.append(segment)

            token_s_pointer += 1
            alignment.token_ids_with_blanks.extend([constants.BLANK_ID])
            alignment.segments_and_tokens.append(BlankToken(s_start=token_s_pointer, s_end=token_s_pointer))

            if i_segment < len(text_segments) - 1:
                token_s_pointer += 1
                alignment.token_ids_with_blanks.extend([constants.SPACE_ID])
                alignment.segments_and_tokens.append(SpaceToken(s_start=token_s_pointer, s_end=token_s_pointer))

                token_s_pointer += 1
                alignment.token_ids_with_blanks.extend([constants.BLANK_ID])
                alignment.segments_and_tokens.append(BlankToken(s_start=token_s_pointer, s_end=token_s_pointer))

        return
