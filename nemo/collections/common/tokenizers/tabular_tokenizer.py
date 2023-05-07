import pickle
from typing import List, Optional

import numpy as np

from column_code import ColumnCodes


END_OF_TEXT = '<|endoftext|>'
NEW_LINE = '\n'


def find_index_of(list_input, item):
    output = -1
    try:
        output = list_input.index(item)
    except ValueError:
        pass
    return output


class TokenizerSpec(object):
    """
    Inherit this class to implement a new tokenizer.
    """

    def encode(self, text: str):
        raise NotImplementedError

    def decode(self, token_ids):
        raise NotImplementedError

    def tokenize(self, text):
        return self.encode(text)

    def detokenize(self, token_ids):
        return self.decode(token_ids)

    def add_special_tokens(self, special_tokens: List[str]):
        raise NotImplementedError("To be implemented")

    @property
    def name(self):
        return type(self).__name__


class TabularTokenizer(TokenizerSpec):
    """
    Composite Tabular Tokenizer allows for multiple tokenizers to be used depending on the table structure
    """

    def __init__(self, coder, special_tokens=None, delimiter=','):
        if special_tokens is None:
            special_tokens = [NEW_LINE, END_OF_TEXT]

        if isinstance(coder, ColumnCodes):
            self.code_column: ColumnCodes = coder
        else:
            with open(coder, 'rb') as handle:
                self.code_column: ColumnCodes = pickle.load(handle)

        self.num_columns = len(self.code_column.sizes)  # len(self.code_column.columns)
        self.special_tokens_encoder = {}
        self.special_tokens_decoder = {}
        self.add_special_tokens(special_tokens)
        self.delimiter = delimiter
        self.eod_id = self.special_tokens_encoder[END_OF_TEXT]
        self.eos_id = self.eod_id
        self.bos_id = self.eos_id
        self.transformed_width = sum(self.code_column.sizes) + 1  # add 1 for newline char each row

        self.partial_row_placeholder = '<PARTIAL_ROW_PLACEHOLDER>'
        self.partial_row_placeholder_token_id = self.vocab_size + 1
        for column in self.code_column.columns:
            if (
                hasattr(self.code_column.column_codes[column], 'special_tokens_to_token_ids')
                and self.partial_row_placeholder in self.code_column.column_codes[column].special_tokens_to_token_ids
            ):
                raise ValueError(f'Cannot use partial_row_placeholder due to conflict as a special token in {column}')

    def __len__(self):
        return self.vocab_size

    @property
    def vocab_size(self):
        return max(self.special_tokens_decoder.keys()) + 1

    @property
    def eod(self):
        return self.eod_id

    @property
    def eor(self):
        return self.special_tokens_encoder[NEW_LINE]

    def text_to_tokens(self, text: str, columns: Optional[List[str]] = None) -> List[int]:
        """
        The table encoding process. Takes a string input, the document, and converts it to a tabular format
        Then maps to token ids, adds newline char, and end of text

        if columns are passed, then we go in the order of the column names
        if partial rows are passed and no columns, we assume the partial row is in the same order as the registered cols

        If a single partial row is passed, no newline char is added as the last char to allow generation of the rest of the row.
        """
        # makes a numpy array o f strings. Now need to convert to the appropriate dtype based on column index
        split = text.split(END_OF_TEXT)
        if len(split) > 2:
            raise ValueError('There are multiple END_OF_TEXT in this document! Please clean docs thoroughly...')
        if len(split) == 2 and split[0] and split[1]:
            raise ValueError(
                'END_OF_TEXT located in the middle of the document. This is not handled yet. ' 'Pass as 2 docs'
            )

        # only add end of text token if it also exists too, hence check for len of split
        end_of_text_position_at_end_of_doc = True if split[0] and len(split) == 2 else False

        max_l = 0
        data = []
        for row in text.replace(END_OF_TEXT, '').split(NEW_LINE):
            d = row.split(self.delimiter)
            max_l = max(len(d), max_l)
            data.append(d)
        for idx in range(len(data)):
            if len(data[idx]) != max_l:  # then we have a partial row
                for iteration in range(max_l - len(data[idx])):
                    data[idx].append(self.partial_row_placeholder)

        data = np.array(data)
        # we don't assert for partial row generation that the number of rows == 1 since we could be interested in
        # generating a subset of the table instead of just the rest of the row.

        if columns and data.shape[1] != len(columns):
            raise ValueError('number of columns does not equal number of fields in passed data')

        # this allows for unordered columns, and/or fewer columns than in the original table. If fewer columns in the
        # original table, it assumes the columns are ordered
        columns = columns if columns else self.code_column.columns[: min(len(self.code_column.columns), data.shape[1])]

        # # add an extra column for the newline chars - this is added at the end?????????
        token_column_width = sum([self.code_column.column_codes[col].code_len for col in columns])
        if data.shape[0] > 1:
            token_column_width += 1  # add 1 for newline char each row

        token_ids = np.empty(shape=(data.shape[0], token_column_width), dtype=int)
        # # Add the newline token to the end of each row
        token_ids[:, -1] = self.special_tokens_encoder[NEW_LINE]

        current_col_idx = 0

        # this assumes no partial row and assumes ordering
        for idx, column_name in enumerate(columns):
            subset = data[:, idx]
            tokens = []
            if self.code_column.column_codes[column_name].__class__.__name__ == 'CategoryCode':
                subset = subset.astype(str)
                for item in subset:
                    if item == self.partial_row_placeholder:
                        tokens.append(
                            self.code_column.column_codes[column_name].code_len
                            * [self.partial_row_placeholder_token_id]
                        )
                        continue
                    tokens.append(self.code_column.column_codes[column_name].encode(item))
                    # tokens.extend(self.code_column.column_codes[column_name].encode(item))
            elif self.code_column.column_codes[column_name].__class__.__name__ in ['IntCode', 'FloatCode']:
                for item in subset:
                    if item == self.partial_row_placeholder:
                        tokens.append(
                            self.code_column.column_codes[column_name].code_len
                            * [self.partial_row_placeholder_token_id]
                        )
                        continue
                    tokens.append(self.code_column.column_codes[column_name].encode(item))
            # elif self.code_column.column_codes[column_name].__class__.__name__ == 'FloatCode':
            #     for item in subset:
            #         tokens.append(self.code_column.column_codes[column_name].encode(item))
            elif self.code_column.column_codes[column_name].__class__.__name__ == 'VectorCode':
                subset = subset.astype(float)
                tokens = self.code_column.column_codes[column_name].encode(subset)
            else:
                raise NotImplementedError('Unable to encode column schema. Is this a new tokenizer?')
            token_ids[
                :, current_col_idx : current_col_idx + self.code_column.column_codes[column_name].code_len
            ] = tokens
            current_col_idx += self.code_column.column_codes[column_name].code_len

        include_last_newline = True
        if token_ids[-1, -2] == self.partial_row_placeholder_token_id:
            include_last_newline = False
        # ravel to a big long list
        token_ids = token_ids[token_ids < self.vocab_size].ravel().tolist()
        if not include_last_newline:
            token_ids = token_ids[:-1]

        if end_of_text_position_at_end_of_doc:
            token_ids = token_ids + [self.special_tokens_encoder[END_OF_TEXT]]
        else:
            token_ids = [self.special_tokens_encoder[END_OF_TEXT]] + token_ids
        return token_ids

    def add_special_tokens(self, special_tokens: List[str]):
        """ Add a list of additional tokens to the encoder.
            The additional tokens are indexed starting from the last
            index of the
            current vocabulary in the order of the `special_tokens` list.
        """
        if special_tokens:

            new = dict(
                (tok, self.code_column.vocab_size + i)
                for i, tok in enumerate(special_tokens)
                if tok not in self.special_tokens_encoder
            )
            self.special_tokens_encoder.update(new)
            self.special_tokens_decoder = {v: k for k, v in self.special_tokens_encoder.items()}

    def encode(self, text: str, columns: Optional[List[str]] = None):
        """takes input string, converts to array or dataframe, and then encodes to tokens"""
        return self.text_to_tokens(text, columns)

    def tokenize(self, text, columns: Optional[List[str]] = None):
        return self.encode(text, columns)

    def token_ids_to_text(self, token_ids: List[int]) -> str:
        """
        Decodes the token_ids to text string. It creates a buffer to hold token_ids if there are multiple token_ids
        required to decode an item. After the buffer is filled, it will decode the token_ids.

        Args:
            token_ids: List of integers representing the token ids

        Returns:
            string of the decoded output incorporating delimiters and adding special tokens if included in the input.

        """
        code_ranges = self.code_column.get_code_ranges()
        decoded = []
        buffer = []
        dec_col_idx = -1
        for token_id in token_ids:
            if buffer:
                if len(buffer) < len(code_ranges[dec_col_idx]):  # then defintely have a new column
                    tok_rng = code_ranges[dec_col_idx][len(buffer)]
                    if tok_rng[0] <= token_id < tok_rng[1]:
                        buffer.append(token_id)
                    else:
                        raise ValueError(
                            f'Cannot decode token_id: {token_id} with column_idx {dec_col_idx} and these'
                            f'ranges: {code_ranges[dec_col_idx]}'
                        )
                else:
                    # decode, add delimiter, and set buffer to [] and dec_col_idx to -1
                    decoded.append(self.code_column.decode(self.code_column.columns[dec_col_idx], buffer))
                    decoded.append(self.delimiter)
                    buffer = []
                    dec_col_idx = -1
            if token_id in self.special_tokens_decoder:
                if token_id == self.special_tokens_encoder[END_OF_TEXT]:
                    decoded.append(str(self.special_tokens_decoder[token_id]))
                elif token_id == self.special_tokens_encoder[NEW_LINE]:
                    # join newline to previous value
                    if decoded[-1] == self.delimiter:
                        decoded.pop()
                    decoded[-1] = decoded[-1] + NEW_LINE
            elif dec_col_idx == -1:
                # this could be improved with binary search
                for col_idx, code_rng in zip(self.code_column.column_idx, code_ranges):
                    for tok_rng in code_rng:
                        if tok_rng[0] <= token_id < tok_rng[1]:
                            buffer.append(token_id)
                            dec_col_idx = col_idx

        return ''.join(decoded).rstrip(NEW_LINE)

    def decode(self, token_ids):
        return self.token_ids_to_text(token_ids)

    def detokenize(self, token_ids):
        return self.decode(token_ids)
