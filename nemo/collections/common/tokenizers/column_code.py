from typing import Dict, List, Tuple, Union

import numpy as np

from code_spec import Code
from datatype_code import IntCode, FloatCode, CategoryCode
# from vector_tokenizer import VectorCode


column_map = {"int": IntCode, "float": FloatCode, "category": CategoryCode}  #, 'vector': VectorCode}


class ColumnCodes(object):
    def __init__(self):
        self.column_codes: Dict[str, Code] = {}
        self.columns = []
        self.column_idx = []
        self.sizes = []

    @property
    def vocab_size(self):
        if not self.columns:
            return 0
        return self.column_codes[self.columns[-1]].end_id

    def register(self, name: Union[str, List[str], Tuple[str]], indices: Union[int, List[int]], ccode: Code):
        """Registers the column name(s), column index ( or list of indicies) and the tokenizer to the class"""
        assert isinstance(name, (str, list, tuple)), 'column name must be a string or list, or tuple of strings'
        if isinstance(name, (list, set)):
            assert all(isinstance(col, str) for col in name), 'all column names must be a string dtype'
            if isinstance(name, list):
                name = tuple(name)
        assert isinstance(indices, (int, list)), 'indices must be int or list of ints'
        if isinstance(indices, list):
            assert all(isinstance(idx, int) for idx in indices), 'all column indices must be int dtype'

        self.columns.append(name)
        self.column_codes[name] = ccode
        self.column_idx.append(indices)  # column index in the table
        # if isinstance(name, (list, tuple)) and isinstance(ccode, VectorCode):
        #     self.sizes.extend([ccode.float_code_len] + (len(name)-1) * [ccode.num_tokens_for_degrees])
        # else:
        self.sizes.append(ccode.code_len)

    def encode(self, col: str, item) -> List[int]:
        """
        Calls a column's encode method by passing in the provided item. The dtype of item is usually a string
        Args:
            col: (str) the column name
            item: the object to be encoded (usually a string)

        Returns: The token id(s)

        """
        if isinstance(col, list):
            col = tuple(col)

        if col in self.column_codes:
            return self.column_codes[col].encode(item)
        else:
            raise ValueError(f"cannot encode {col} {item}")

    def decode(self, col: str, ids: List[int]) -> str:
        """
        Call's a columns decode method by passing in the provided token ids.
        Args:
            col: (str) the column name
            ids: the list of token ids to be decoded

        Returns: the decoded object, usually a string.

        """
        if col in self.column_codes:
            return self.column_codes[col].decode(ids)
        else:
            raise ValueError(f"cannot decode column {col} with token_ids {ids} because it has not been added via the "
                             f"'register' method")

    def get_range(self, column_id: int) -> List[Tuple[int, int]]:
        """Get single column's token_id range"""
        return self.column_codes[self.columns[column_id]].code_range

    def get_code_ranges(self) -> List[List[Tuple[int, int]]]:
        """
        Get all the column's token_id ranges
        Returns:

        """
        return [self.get_range(col_idx) for col_idx in self.column_idx]

    @classmethod
    def get_column_codes(cls, column_configs, example_arrays):
        """
        Trains a ColumnCodes tokenizer using the provided configs and example arrays.
        Args:
            column_configs: list of dicts
            example_arrays: list of numpy arrays

        Returns:

        """
        column_codes = cls()
        beg = 0
        cc = None
        for config, example_array in zip(column_configs, example_arrays):
            col_name = config['name']
            coder = column_map[config['code_type']]
            indices = config['idx']
            # if isinstance(indices, list) and np.diff(indices).max() > 1:
            #     raise ValueError('Vector Columns must be contiguous. Please reorder columns and try again.')
            args = config.get('args', {})
            start_id = beg if cc is None else cc.end_id
            # args['start_id'] = start_id
            # args['col_name'] = col_name
            cc = coder(col_name=col_name, start_id=start_id, **args)
            cc.compute_code(example_array)

            column_codes.register(col_name, indices, cc)
        return column_codes
