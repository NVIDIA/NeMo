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

import os
import re
import string

import numpy as np

from nemo import logging

__all__ = [
    '_is_whitespace',
    'mask_padded_tokens',
    'read_intent_slot_outputs',
    'get_vocab',
    'write_vocab',
    'label2idx',
    'write_vocab_in_order',
    'if_exist',
    'remove_punctuation_from_sentence',
    'ids2text',
    'calc_class_weights',
]


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def mask_padded_tokens(tokens, pad_id):
    mask = tokens != pad_id
    return mask


def read_intent_slot_outputs(
    queries, intent_file, slot_file, intent_logits, slot_logits, slot_masks, intents=None, slots=None
):
    intent_dict = get_vocab(intent_file)
    slot_dict = get_vocab(slot_file)
    pred_intents = np.argmax(intent_logits, 1)
    pred_slots = np.argmax(slot_logits, axis=2)
    slot_masks = slot_masks > 0.5
    for i, query in enumerate(queries):
        logging.info(f'Query: {query}')
        pred = pred_intents[i]
        logging.info(f'Predicted intent:\t{pred}\t{intent_dict[pred]}')
        if intents is not None:
            logging.info(f'True intent:\t{intents[i]}\t{intent_dict[intents[i]]}')

        pred_slot = pred_slots[i][slot_masks[i]]
        tokens = query.strip().split()

        if len(pred_slot) != len(tokens):
            raise ValueError('Pred_slot and tokens must be of the same length')

        for j, token in enumerate(tokens):
            output = f'{token}\t{slot_dict[pred_slot[j]]}'
            if slots is not None:
                output = f'{output}\t{slot_dict[slots[i][j]]}'
            logging.info(output)


def get_vocab(file):
    lines = open(file, 'r').readlines()
    lines = [line.strip() for line in lines if line.strip()]
    labels = {i: lines[i] for i in range(len(lines))}
    return labels


def write_vocab(items, outfile):
    vocab = {}
    idx = 0
    with open(outfile, 'w') as f:
        for item in items:
            f.write(item + '\n')
            vocab[item] = idx
            idx += 1
    return vocab


def label2idx(file):
    lines = open(file, 'r').readlines()
    lines = [line.strip() for line in lines if line.strip()]
    labels = {lines[i]: i for i in range(len(lines))}
    return labels


def write_vocab_in_order(vocab, outfile):
    with open(outfile, 'w') as f:
        for key in sorted(vocab.keys()):
            f.write(f'{vocab[key]}\n')


def if_exist(outfold, files):
    if not os.path.exists(outfold):
        return False
    for file in files:
        if not os.path.exists(f'{outfold}/{file}'):
            return False
    return True


def remove_punctuation_from_sentence(sentence):
    sentence = re.sub('[' + string.punctuation + ']', '', sentence)
    sentence = sentence.lower()
    return sentence


def ids2text(ids, vocab):
    return ' '.join([vocab[int(id_)] for id_ in ids])


def calc_class_weights(label_freq):
    """
    Goal is to give more weight to the classes with less samples
    so as to match the one with the higest frequency. We achieve this by
    dividing the highest frequency by the freq of each label.
    Example -
    [12, 5, 3] -> [12/12, 12/5, 12/3] -> [1, 2.4, 4]

    Here label_freq is assumed to be sorted by the frequency. I.e.
    label_freq[0] is the most frequent element.

    """

    most_common_label_freq = label_freq[0]
    weighted_slots = sorted([(index, most_common_label_freq[1] / freq) for (index, freq) in label_freq])
    return [weight for (_, weight) in weighted_slots]
