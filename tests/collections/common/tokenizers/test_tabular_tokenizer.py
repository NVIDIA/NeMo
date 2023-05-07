import numpy as np
import pandas as pd
import pytest
from tabular_tokenizer import TabularTokenizer

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 100)

END_OF_TEXT = '<|endoftext|>'
NEW_LINE = '\n'
DELIMITER = ','


@pytest.fixture()
def tabular_tokenizer(column_codes):
    _tabular_tokenizer = TabularTokenizer(column_codes, special_tokens=[NEW_LINE, END_OF_TEXT], delimiter=DELIMITER)
    return _tabular_tokenizer


@pytest.fixture()
def doc(data):
    # can parameterize this later.
    u = data['0_categorical_letter'].astype(str).copy()
    for col in data.columns[1:5]:
        u = u.str.cat(data[col].astype(str), sep=',')
    doc = {'text': u.str.cat(sep='\n')}
    return doc


def test_add_special_tokens():
    # assert False
    pass


def test_text_to_tokens(tabular_tokenizer, corpus, doc, data):
    """

    The only reason text_to_tokens, encode, and tokenize are all the same is for compatibility reasons.

    Similar with the token_ids_to_text,
    """
    u = data['0_categorical_letter'].astype(str).copy()
    for col in data.columns[1:5]:
        u = u.str.cat(data[col].astype(str), sep=',')
    doc = {'text': u.str.cat(sep='\n')}
    token_ids = tabular_tokenizer.tokenize(doc['text'])
    # token_ids = tabular_tokenizer.tokenize(corpus[0]['text'])
    assert isinstance(token_ids, list) and token_ids, 'token_ids is a non-empty list of integers'
    assert all(isinstance(i, int) for i in token_ids), 'all token ids should be integers'
    assert all(i <= tabular_tokenizer.vocab_size for i in token_ids), 'all token ids should be less than vocab size'
    assert (len(token_ids) - 1) % token_ids.index(tabular_tokenizer.special_tokens_encoder['\n']) == 0

    u = data['0_categorical_letter'].astype(str).copy()
    for col in data.columns[5:]:
        u = u.str.cat(data[col].astype(str), sep=',')
    doc = {'text': u.str.cat(sep='\n')}
    token_ids = tabular_tokenizer.tokenize(doc['text'], [data.columns[0]] + list(data.columns[5:]))
    assert all(i <= tabular_tokenizer.vocab_size for i in token_ids), 'all token ids should be integers'
    assert (len(token_ids) - 1) % token_ids.index(tabular_tokenizer.special_tokens_encoder['\n']) == 0

    # pass just a partial row with unordered cols
    u = data['0_categorical_letter'].astype(str).copy()
    for col in data.columns[5:]:
        u = u.str.cat(data[col].astype(str), sep=',')
    doc = {'text': u.iloc[0]}
    token_ids = tabular_tokenizer.tokenize(doc['text'], [data.columns[0]] + list(data.columns[5:]))
    assert all(i <= tabular_tokenizer.vocab_size for i in token_ids), 'all token ids should be integers'

    # passes 2 rows with unordered cols where the second row is a partial row missing the last 2 items
    u = data['0_categorical_letter'].astype(str).copy()
    for col in data.columns[5:]:
        u = u.str.cat(data[col].astype(str), sep=',')
    doc = {'text': u.iloc[:2].str.cat(sep='\n').rsplit(',', 2)[0]}
    token_ids = tabular_tokenizer.tokenize(doc['text'], [data.columns[0]] + list(data.columns[5:]))
    assert all(i <= tabular_tokenizer.vocab_size for i in token_ids), 'all token ids should be integers'
    # check newline exists but is not last token
    assert (
        tabular_tokenizer.special_tokens_encoder['\n'] in token_ids
        and token_ids[-1] != tabular_tokenizer.special_tokens_encoder['\n']
    )
    # ensure that for partial row that the rows don't all have the same elements. Subtract 1 because of endoftext token
    assert (len(token_ids) - 1) % token_ids.index(tabular_tokenizer.special_tokens_encoder['\n']) != 0


def test_encode(tabular_tokenizer, doc):
    token_ids = tabular_tokenizer.tokenize(doc['text'])
    assert isinstance(token_ids, list) and token_ids, 'token_ids is a non-empty list of integers'
    assert all(isinstance(i, int) for i in token_ids), 'all token ids should be integers'


def test_tokenize(tabular_tokenizer, doc):
    token_ids = tabular_tokenizer.tokenize(doc['text'])
    assert isinstance(token_ids, list) and token_ids, 'token_ids is a non-empty list of integers'
    assert all(isinstance(i, int) for i in token_ids), 'all token ids should be integers'


def test_token_ids_to_text(tabular_tokenizer, data):
    u = data['0_categorical_letter'].astype(str).copy()
    for col in data.columns[1:5]:
        u = u.str.cat(data[col].astype(str), sep=',')
    doc = {'text': u.str.cat(sep='\n')}
    token_ids = tabular_tokenizer.tokenize(doc['text'])

    # document = corpus[0]['text']
    # token_ids = tabular_tokenizer.text_to_tokens(document)
    output = tabular_tokenizer.token_ids_to_text(token_ids)

    u = data['0_categorical_letter'].astype(str).copy()
    for col in data.columns[5:]:
        u = u.str.cat(data[col].astype(str), sep=',')
    doc = {'text': u.str.cat(sep='\n')}
    token_ids = tabular_tokenizer.tokenize(doc['text'], [data.columns[0]] + list(data.columns[5:]))
    output = tabular_tokenizer.token_ids_to_text(token_ids)

    # pass just a partial row with unordered cols
    u = data['0_categorical_letter'].astype(str).copy()
    for col in data.columns[5:]:
        u = u.str.cat(data[col].astype(str), sep=',')
    doc = {'text': u.iloc[0]}
    token_ids = tabular_tokenizer.tokenize(doc['text'], [data.columns[0]] + list(data.columns[5:]))
    output = tabular_tokenizer.token_ids_to_text(token_ids)

    # passes 2 rows with unordered cols where the second row is a partial row missing the last 2 items
    u = data['0_categorical_letter'].astype(str).copy()
    for col in data.columns[5:]:
        u = u.str.cat(data[col].astype(str), sep=',')
    doc = {'text': u.iloc[:2].str.cat(sep='\n').rsplit(',', 2)[0]}
    token_ids = tabular_tokenizer.tokenize(doc['text'], [data.columns[0]] + list(data.columns[5:]))
    output = tabular_tokenizer.token_ids_to_text(token_ids)


def test_decode(tabular_tokenizer, doc):
    document = doc['text']
    token_ids = tabular_tokenizer.text_to_tokens(document)
    tabular_tokenizer.decode(token_ids)


def test_detokenize(tabular_tokenizer, doc):
    document = doc['text']
    token_ids = tabular_tokenizer.text_to_tokens(document)
    tabular_tokenizer.detokenize(token_ids)
