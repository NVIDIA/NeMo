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
import string
from copy import deepcopy

from nltk import word_tokenize
from tqdm import tqdm

__all__ = ['read_data_file', 'normalize_str']


def read_data_file(fp):
    """ Reading the raw data from a file of NeMo format
    For more info about the data format, refer to the
    `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`.
    """
    insts, w_words, s_words, classes = [], [], [], []
    # Read input file
    with open(fp, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            es = [e.strip() for e in line.strip().split('\t')]
            if es[0] == '<eos>':
                inst = (deepcopy(classes), deepcopy(w_words), deepcopy(s_words))
                insts.append(inst)
                # Reset
                w_words, s_words, classes = [], [], []
            else:
                classes.append(es[0])
                w_words.append(es[1])
                s_words.append(es[2])
    return insts


def normalize_str(input_str):
    """ Normalize an input string """
    input_str = ' '.join(word_tokenize(input_str.strip().lower()))
    input_str = input_str.replace('  ', ' ')
    return input_str


def remove_puncts(input_str):
    """ Remove punctuations from an input string """
    return input_str.translate(str.maketrans('', '', string.punctuation))
