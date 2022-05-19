# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from pydoc import doc

import pytest

from nemo.collections.nlp.data.question_answering_squad.qa_dataset import SquadDataset
from nemo.collections.nlp.data.question_answering_squad.qa_squad_processing import (
    _get_tokens,
    exact_match_score,
    f1_score,
)


@pytest.mark.unit
def test_get_tokens():
    sentence = 'I am happy'
    tokens = ['i', 'am', 'happy']
    assert tokens == _get_tokens(sentence)

    sentence = 'I am a person'
    tokens = ['i', 'am', 'person']
    assert tokens == _get_tokens(sentence)

    sentence = 'I am a person.'
    tokens = ['i', 'am', 'person']
    assert tokens == _get_tokens(sentence)


@pytest.mark.unit
def test_f1_score():

    generated_field = 'That is so good'
    ground_truth_field = 'That is so awesome'

    f1 = f1_score(generated_field, ground_truth_field)
    assert f1 == 0.75

    generated_field = ''
    ground_truth_field = 'That'

    f1 = f1_score(generated_field, ground_truth_field)
    assert f1 == 0


@pytest.mark.unit
def test_exact_match_score():

    generated_field = 'That is so good'
    ground_truth_field = 'That is so awesome'

    em = exact_match_score(generated_field, ground_truth_field)
    assert em == 0

    generated_field = 'That is so good!'
    ground_truth_field = 'That is so good.'

    em = exact_match_score(generated_field, ground_truth_field)
    assert em == 1

    generated_field = 'That is so good'
    ground_truth_field = 'that is so good'

    em = exact_match_score(generated_field, ground_truth_field)
    assert em == 1


@pytest.mark.unit
def test_split_into_words():
    text = 'hi yo'
    char_to_word_offset = [0, 0, 0, 1, 1]
    doc_tokens = ["hi", "yo"]
    output = SquadDataset.split_into_words(text)
    assert output[0] == doc_tokens
    assert output[1] == char_to_word_offset

    text = 'i am good'
    char_to_word_offset = [0, 0, 1, 1, 1, 2, 2, 2, 2]
    doc_tokens = ["i", "am", 'good']
    output = SquadDataset.split_into_words(text)
    assert output[0] == doc_tokens
    assert output[1] == char_to_word_offset


@pytest.mark.unit
def test_get_doc_spans():
    all_doc_tokens = ['a'] * 15
    max_tokens_for_doc = 10
    doc_stride = 5
    doc_spans = SquadDataset.get_docspans(all_doc_tokens, max_tokens_for_doc, doc_stride)

    assert len(doc_spans) == 2
    assert doc_spans[0].start == 0
    assert doc_spans[0].length == 10
    assert doc_spans[1].start == 5
    assert doc_spans[1].length == 10
