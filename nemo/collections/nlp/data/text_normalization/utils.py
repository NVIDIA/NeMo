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
import re
import string
from copy import deepcopy
from typing import List

import wordninja
from nltk import word_tokenize
from tqdm import tqdm

from nemo.collections.nlp.data.text_normalization import constants
from nemo.utils import logging

__all__ = ['read_data_file', 'normalize_str', 'flatten', 'process_url']


def flatten(l):
    """ flatten a list of lists """
    return [item for sublist in l for item in sublist]


def input_preprocessing(sent: str, lang: str):
    """ Function for preprocessing the input texts. The function first does
    some basic tokenization. For English, it then also processes Greek letters
    such as Δ or λ (if any).

    Args:
        sents: input text.
        lang: language

    Returns: preprocessed input text.
    """
    # Basic Preprocessing and Tokenization
    if lang == constants.ENGLISH:
        sent = sent.replace('+', ' plus ')
        sent = sent.replace('=', ' equals ')
        sent = sent.replace('@', ' at ')
        sent = sent.replace('*', ' times ')

    # Greek letters processing
    if lang == constants.ENGLISH:
        for jx, tok in enumerate(sent):
            if tok in constants.EN_GREEK_TO_SPOKEN:
                sent = sent[:jx] + constants.EN_GREEK_TO_SPOKEN[tok] + sent[jx + 1 :]
    return sent


def read_data_file(fp: str, lang: str, max_insts: int = -1):
    """ Reading the raw data from a file of NeMo format
    For more info about the data format, refer to the
    `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`.

    Args:
        fp: file paths
        lang: language
        max_insts: Maximum number of instances (-1 means no limit)
    Returns:
        insts: List of sentences parsed as list of words
    """
    insts, w_words, s_words, classes = [], [], [], []
    # Read input file
    with open(fp, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            es = [e.strip() for e in input_preprocessing(line.strip(), lang=lang).split('\t')]
            if es[0] == '<eos>':
                inst = (deepcopy(classes), deepcopy(w_words), deepcopy(s_words))
                insts.append(inst)
                # Reset
                w_words, s_words, classes = [], [], []

                if max_insts > 0 and len(insts) >= max_insts:
                    break
            else:
                classes.append(es[0])
                w_words.append(es[1])
                s_words.append(es[2])
    return insts


def process_url(tokens: List[str], outputs: List[str], lang: str):
    """
    The function is used to process the spoken form of every URL in an example.
    E.g., "dot h_letter  _letter t_letter  _letter m_letter  _letter l_letter" ->
          "dot h t m l"

    Args:
        tokens: The tokens of the written form
        outputs: The expected outputs for the spoken form
        lang: Selected language.
    Return:
        outputs: The outputs for the spoken form with preprocessed URLs.
    """
    if lang != constants.ENGLISH:
        return outputs

    for i in range(len(tokens)):
        t, o = tokens[i], outputs[i]
        if o != constants.SIL_WORD and '_letter' in o:
            o_tokens = o.split(' ')
            all_spans, cur_span = [], []
            for j in range(len(o_tokens)):
                if len(o_tokens[j]) == 0:
                    continue
                if o_tokens[j] == '_letter':
                    all_spans.append(cur_span)
                    all_spans.append([' '])
                    cur_span = []
                else:
                    o_tokens[j] = o_tokens[j].replace('_letter', '')
                    cur_span.append(o_tokens[j])
            if len(cur_span) > 0:
                all_spans.append(cur_span)
            o_tokens = flatten(all_spans)

            o = ''
            for o_token in o_tokens:
                if len(o_token) > 1:
                    o += ' ' + o_token + ' '
                else:
                    o += o_token
            o = o.strip()
            o_tokens = wordninja.split(o)
            o = ' '.join(o_tokens)

            outputs[i] = o
    return outputs


def normalize_str(input_str, lang):
    """ Normalize an input string """
    input_str_tokens = basic_tokenize(input_str.strip().lower(), lang)
    input_str = ' '.join(input_str_tokens)
    input_str = input_str.replace('  ', ' ')
    return input_str


def remove_puncts(input_str):
    """ Remove punctuations from an input string """
    return input_str.translate(str.maketrans('', '', string.punctuation))


def basic_tokenize(input_str, lang):
    """
    The function is used to do some basic tokenization

    Args:
        input_str: The input string
        lang: Language of the input string
    Return: a list of tokens of the input string
    """
    if lang == constants.ENGLISH:
        return word_tokenize(input_str)
    return input_str.strip().split(' ')


def post_process_punct(input: str, nn_output: str):
    """
    Post-processing of the normalized output to match input in terms of spaces around punctuation marks.
    After NN normalization, Moses detokenization puts a space after
    punctuation marks, and attaches an opening quote "'" to the word to the right.
    E.g., input to the TN NN model is "12 test' example",
    after normalization and detokenization -> "twelve test 'example" (the quote is considered to be an opening quote,
    but it doesn't match the input and can cause issues during TTS voice generation.)
    The current function will match the punctuation and spaces of the normalized text with the input sequence.
    "12 test' example" -> "twelve test 'example" -> "twelve test' example" (the quote was shifted to match the input).

    Args:
        input: input text (original input to the NN, before normalization or tokenization)
        nn_output: output text (output of the TN NN model)
    """
    input = [x for x in input]
    nn_output = [x for x in nn_output]
    punct_marks = string.punctuation

    try:
        for punct in punct_marks:
            if input.count(punct) != nn_output.count(punct):
                continue
            idx_in, idx_out = 0, 0
            while punct in input[idx_in:]:
                idx_in = input.index(punct, idx_in)
                idx_out = nn_output.index(punct, idx_out)
                if idx_in > 0 and idx_out > 0:
                    if nn_output[idx_out - 1] == " " and input[idx_in - 1] != " ":
                        nn_output[idx_out - 1] = ""

                    elif nn_output[idx_out - 1] != " " and input[idx_in - 1] == " ":
                        nn_output[idx_out - 1] += " "

                if idx_in < len(input) - 1 and idx_out < len(nn_output) - 1:
                    if nn_output[idx_out + 1] == " " and input[idx_in + 1] != " ":
                        nn_output[idx_out + 1] = ""
                    elif nn_output[idx_out + 1] != " " and input[idx_in + 1] == " ":
                        nn_output[idx_out] = nn_output[idx_out] + " "
                idx_out += 1
                idx_in += 1
    except:
        logging.warning(f"Skipping post-processing of {''.join(nn_output)}")
    nn_output = "".join(nn_output)
    return re.sub(r' +', ' ', nn_output)
