import pytest
import numpy as np
import math

from tqdm import tqdm

from datatype_code import FloatCode, IntCode


@pytest.fixture()
def data_series():
    return np.linspace(0, 360, 360 * 10 ** 1)


@pytest.fixture()
def float_code(data_series):
    float_cd = FloatCode('name', code_len=3, start_id=0, fill_all=True, base=100, special_tokens=['a', 'b'],
                         has_nan=True, transform='best')
    float_cd.compute_code(data_series)
    return float_cd


def test_compute_code(data_series, float_code):
    float_code.compute_code(data_series)
    assert len(float_code.digits_item_to_id) == len(float_code.digits_id_to_item) == float_code.code_len
    assert float_code.min_value != -np.inf


def test_transform_data_to_int_array(data_series, float_code):
    out = float_code.transform_data_to_int_array(data_series)
    digits = int(math.log(data_series.max(), float_code.base) + 1)
    assert float_code.extra_digits == float_code.code_len - digits


def test_encode(data_series, float_code):
    encoded = []
    for item in np.random.choice(data_series.astype(str), 1000):
        encoded.append(float_code.encode(item))
    float_code.encode('a')
    assert all([len(i) == float_code.code_len for i in encoded])


def test_decode(data_series, float_code):
    decoded = []
    to_encode = np.random.choice(data_series, 1000)
    for item in to_encode:
        decoded.append(float(float_code.decode(float_code.encode(str(item)))))
    decoded = np.array(decoded)
    a = float_code.decode(float_code.encode('a'))
    b = float_code.decode(float_code.encode('b'))
    nan = float_code.decode(float_code.encode('nan'))
    assert a == 'a' and b == 'b' and nan == 'nan'

    assert max(abs(decoded-to_encode)/abs(to_encode)) < 0.05
