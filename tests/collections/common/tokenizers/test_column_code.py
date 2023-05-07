import numpy as np
import pytest

from column_code import ColumnCodes


def test_vocab_size(column_codes):
    assert column_codes.vocab_size == column_codes.column_codes[column_codes.columns[-1]].end_id
    assert ColumnCodes().vocab_size == 0


def test_encode(column_codes, data):
    for col in column_codes.columns:
        if isinstance(col, str):
            item = data[col].iloc[0]
            item = item.astype(str) if not isinstance(item, str) else item
            column_codes.column_codes[col].encode(item)
        # # for vector based tokenization.
        # elif isinstance(col, (tuple, list)):
        #     select_col = list(col) if isinstance(col, tuple) else col
        #     # df[tuple([list, of items]) != df[[ list, of items]] as pandas interprets the tuple as a
        #     # single key/column
        #     column_codes.column_codes[col].encode(data[select_col])


def test_get_range(column_codes):
    for idx in range(len(column_codes.column_idx)):
        assert column_codes.get_range(idx)
    with pytest.raises(IndexError):
        assert column_codes.get_range(len(column_codes.column_idx))


def test_get_column_codes(tab_structure, data):
    # example array
    will_error_out = False  # will error out if vector column indices are not contiguous.
    example_arrays = []
    for idx, col in enumerate(tab_structure):
        if col['code_type'] == 'category':
            example_arrays.append(data[col['name']].unique().astype(str))
        elif col['code_type'] in ['int', 'float']:
            example_arrays.append(data[col['name']].dropna().unique())
            if col['code_type'] == 'float' and col['name'] == 'VIX_LEVEL':
                will_error_out = True
        elif col['code_type'] == 'vector':
            example_arrays.append(data[col['name']])
        else:
            raise TypeError('Code_type for col must be one of "float", "int", "category", or "vector"')
    # now the tab_structure and example_arrays get passed

    if will_error_out:
        with pytest.raises(ValueError):
            cc = ColumnCodes.get_column_codes(tab_structure, example_arrays)
    else:
        cc = ColumnCodes.get_column_codes(tab_structure, example_arrays)
        encode_decode = []
        for item in example_arrays[-1]:
            item = item.astype(str) if not isinstance(item, str) else item
            encode_decode.append(cc.column_codes[cc.columns[-1]].decode(cc.column_codes[cc.columns[-1]].encode(item)))
        # VISUALLY INSPECT THIS, since the precision is dependent on the precision set for the numerical tokenizer.
        print(encode_decode, example_arrays[-1])
        print(example_arrays[-1] == encode_decode)
        # below only helpful for float arrays
        # print(np.isclose(encode_decode,
        #                  example_arrays[-1],
        #                  rtol=1e-3).sum(axis=1).argmin()
        #       )
        assert True
