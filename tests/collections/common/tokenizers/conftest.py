import json
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import pytest
from column_code import ColumnCodes
from numpy.random import default_rng

mus = [0.01, 10_000]
sigmas = [1, 200]
Ns = [4000, 4000]


def get_distr_args(rng, dist: str) -> Iterable:
    if dist in ['normal', 'laplace']:  # loc, scale or mean,sigma
        loc = rng.uniform(-10, 10)
        scale = rng.uniform(1e-2, 20)
        return loc, scale
    elif dist == 'lognormal':
        mean = rng.uniform(-3, 3)
        sigma = rng.uniform(0.5, 3)
        return mean, sigma
    elif dist in ['poisson', 't', 'students-t']:
        return [rng.integers(1, 30)]
    elif dist in ['power', 'exponential', 'exp']:
        return [rng.uniform(1e-2, 10)]
    return []


def generate_single_dist(rng, N, dist, *args):

    if dist == 'normal':
        X = rng.normal(*args, size=N)  # loc, scale
    elif dist == 'lognormal':
        X = rng.lognormal(*args, size=N)  # mean, sigma
    elif dist == 'laplace':
        X = rng.laplace(*args, size=N)  # loc, scale
    elif dist == 'exponential' or dist == 'exp':
        X = rng.exponential(*args, size=N)  # scale = 1
    elif dist == 'poisson':
        X = rng.poisson(*args, size=N)  # lam
    elif dist == 'power':
        X = rng.power(*args, size=N)  # 'a' is exponent, a>0
    elif dist == 'cauchy' or dist == 'lorentz':
        X = rng.standard_cauchy(size=N)
    elif dist == 't' or dist == 'students-t':
        X = rng.standard_t(*args, size=N)  # df > 0
    else:
        raise ValueError(
            'dist not found in supported dists: normal, lognormal, laplace, exponential, poisson, power,' ' cauchy, t'
        )
    shift_distr = rng.integers(0, 2)
    if shift_distr:
        shift = rng.uniform(-20, 20)
        if X.dtype == int:
            X += int(shift)
        else:
            X += shift
    return X


def generate_multimodal_data(rng, nrows: int, dists: list):
    assert len(dists) > 1
    assert nrows // len(dists) > 30, 'generate more rows'

    count = 0
    column_data = []
    for dist in dists:
        args = get_distr_args(rng, dist)
        X_i = generate_single_dist(rng, nrows // len(dists), dist, *args)
        count += len(X_i)
        column_data.append(X_i)

    if count != nrows:
        dist = dists[-1]
        args = get_distr_args(rng, dist)
        X_i = generate_single_dist(rng, nrows - count, dist, *args)
        count += len(X_i)
        column_data.append(X_i)
    # background = rng.uniform(min(mus), max(mus), size=max(max(Ns)//5, 1))
    # column_data.append(background)
    X = np.concatenate(column_data)
    return X


def generate_categorical(rng, nrows: int, generate_letters: bool):
    # generate categorical features
    if generate_letters:
        array = rng.choice(list('abcdefghijklmnopqrstuvwxyz'), size=nrows)
    else:
        array = rng.choice(range(100), size=nrows)
    return array


@pytest.fixture(params=[(2000, 10)])
def data(request) -> pd.DataFrame:
    rng = default_rng(seed=0)
    nrows, ncols = request.param
    assert isinstance(nrows, int) and nrows > 0
    assert isinstance(ncols, int) and ncols >= 4, (
        'we want to use one of each column type first, the rest will be ' 'randomly chosen'
    )
    column_options = ['categorical_letter', 'categorical_integer', 'float_single_dist', 'float_multi_dist']
    distribution_options = ['normal', 'lognormal', 'laplace', 'exponential', 'poisson', 'power', 'cauchy', 't']

    options = rng.choice(column_options, size=ncols - len(column_options))
    columns = np.concatenate([column_options, options])

    data = {str(i) + '_' + columns[i]: [] for i in range(len(columns))}

    # loop over columns and generate the rows

    for col_name, column_type in zip(data, columns):
        if column_type == 'categorical_letter':
            array = generate_categorical(rng, nrows, generate_letters=True)
        elif column_type == 'categorical_integer':
            array = generate_categorical(rng, nrows, generate_letters=False)
        elif column_type == 'float_single_dist':
            dist = rng.choice(distribution_options)
            args = get_distr_args(rng, dist)
            array = generate_single_dist(rng, nrows, dist, *args)
        elif column_type == 'float_multi_dist':
            dists = rng.choice(distribution_options, size=rng.integers(2, 10))
            array = generate_multimodal_data(rng, max(nrows, 30 * len(dists)), dists)
        else:
            raise TypeError()
        data[col_name] = array
    return pd.DataFrame(data)


@pytest.fixture()
def tab_structure(data, request):
    columns = data.columns
    tab_struct = []
    # picking a random float col to stay a float col, not vector type
    date_cols = []
    float_cols = [col for col in columns if 'float' in col]  # request.param
    categorical_columns = [col for col in columns if 'letter' in col]
    integer_columns = [col for col in columns if 'integer' in col]
    vector_columns = []  # [col for col in data_cols if col not in float_col]
    vector_cols_structure = {
        'name': [],
        'idx': [],
        "code_type": "vector",
        # all the vector args should be the same for all vector columns
        "args": {
            "radius_code_len": 5,  # number of tokens used to code the column
            "degrees_precision": 'four_decimal',
            'custom_degrees_tokens': 0,
            "base": 100,  # the positional base number. ie. it uses 32 tokens for one digit
            "fillall": True,
            # whether to use full base number for each token or derive it from the data.
            "hasnan": True,  # can it handles nan or not
        },
    }
    for idx, c in enumerate([col for col in columns if col != 'year']):
        item = {}

        if c in date_cols:
            item = {
                "name": c,
                'idx': idx,
                "code_type": "category",
            }
        elif c in categorical_columns:
            item = {
                "name": c,
                'idx': idx,
                "code_type": "category",
            }
        elif c in float_cols:
            item = {
                "name": c,
                'idx': idx,
                "code_type": "float",
                "args": {
                    "code_len": 4,  # number of tokens used to code the column
                    "base": 70,  # the positional base number. i.e. it uses 32 tokens for one digit
                    "fill_all": True,  # whether to use full base number for each token or derive it from the data.
                    "has_nan": True,  # can it handles nan or not
                    "transform": "best"
                    # can be ['yeo-johnson', 'quantile', 'robust', 'log1p', 'identity'], check https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
                },
            }
        elif c in integer_columns:
            item = {
                "name": c,
                "code_type": "int",
                "idx": idx,
                "args": {
                    "code_len": 3,  # number of tokens used to code the column
                    "base": 100,  # the positional base number. ie. it uses 32 tokens for one digit
                    "fill_all": True,  # whether to use full base number for each token or derive it from the data.
                    "has_nan": True,  # can it handles nan or not
                },
            }
        elif c in vector_columns:
            vector_cols_structure['name'].append(c)
            vector_cols_structure['idx'].append(idx)
        if item:
            tab_struct.append(item)
    if vector_columns:
        tab_struct.append(vector_cols_structure)
    return tab_struct


@pytest.fixture()
def column_codes(tab_structure, data):
    """Already instantiated class"""
    columns = data.columns
    df = data

    example_arrays = []
    for idx, col in enumerate(tab_structure):
        if col['code_type'] == 'category':
            example_arrays.append(df[col['name']].unique())
        elif col['code_type'] in ['int', 'float']:
            example_arrays.append(df[col['name']].dropna().unique())
        elif col['code_type'] == 'vector':
            example_arrays.append(df[col['name']])
        else:
            raise TypeError('Code_type for col must be one of "float", "int", "category", or "vector"')
    cc = ColumnCodes.get_column_codes(tab_structure, example_arrays)
    return cc
