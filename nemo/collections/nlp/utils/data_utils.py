# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

import re
import string

import numpy as np

__all__ = ['get_vocab', 'get_tokens', 'normalize_answer', 'mask_padded_tokens', 'concatenate']


def get_vocab(file):
    lines = open(file, 'r').readlines()
    lines = [line.strip() for line in lines if line.strip()]
    labels = {i: lines[i] for i in range(len(lines))}
    return labels


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def mask_padded_tokens(tokens, pad_id):
    mask = tokens != pad_id
    return mask


def concatenate(lists):
    """
    Helper function for inference
    """
    return np.concatenate([t.cpu() for t in lists])
