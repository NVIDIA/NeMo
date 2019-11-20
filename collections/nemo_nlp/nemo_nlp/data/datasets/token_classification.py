# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
"""
Utility functions for Token Classification NLP tasks
Some parts of this code were adapted from the HuggingFace library at
https://github.com/huggingface/pytorch-pretrained-BERT
"""

import itertools
import random

import numpy as np
from torch.utils.data import Dataset

from nemo.utils.exp_logging import get_logger

from . import utils


logger = get_logger('')


def get_features(queries,
                 max_seq_length,
                 tokenizer,
                 pad_label='O',
                 raw_slots=None,
                 all_labels=None,
                 ignore_extra_tokens=False,
                 ignore_start_end=False):

    all_subtokens = []
    all_loss_mask = []
    all_subtokens_mask = []
    all_segment_ids = []
    all_input_ids = []
    all_input_mask = []
    sent_lengths = []
    all_slots = []
    with_label = False

    # Create mapping of labels to label ids that starts with pad_label->0 and
    # then increases in alphabetical order
    label_ids = {pad_label: 0} if raw_slots else None

    if raw_slots is not None:
        with_label = True

        # add pad_label to the set of all_labels if not already present
        all_labels.add(pad_label)

        all_labels.remove(pad_label)
        for label in sorted(all_labels):
            label_ids[label] = len(label_ids)

    for i, query in enumerate(queries):
        words = query.strip().split()
        
        # add bos token
        subtokens = ['[CLS]']
        loss_mask = [not ignore_start_end]
        subtokens_mask = [False]
        if with_label:
            pad_id = label_ids[pad_label]
            slots = [pad_id]
            query_labels = [label_ids[slot] for slot in raw_slots[i]]

        for j, word in enumerate(words):
            word_tokens = tokenizer.text_to_tokens(word)
            subtokens.extend(word_tokens)
            
            loss_mask.append(True)
            loss_mask.extend([not ignore_extra_tokens] *
                             (len(word_tokens) - 1))

            subtokens_mask.append(True)
            subtokens_mask.extend([False] * (len(word_tokens) - 1))

            if with_label:
                slots.extend([query_labels[j]] * len(word_tokens))

        subtokens.append('[SEP]')
        loss_mask.append(not ignore_start_end)
        subtokens_mask.append(False)
        sent_lengths.append(len(subtokens))
        all_subtokens.append(subtokens)
        all_loss_mask.append(loss_mask)
        all_subtokens_mask.append(subtokens_mask)
        all_input_mask.append([1] * len(subtokens))

        if with_label:
            slots.append(pad_id)
            all_slots.append(slots)

    max_seq_length = min(max_seq_length, max(sent_lengths))
    logger.info(f'Max length: {max_seq_length}')
    utils.get_stats(sent_lengths)
    too_long_count = 0

    for i, subtokens in enumerate(all_subtokens):
        if len(subtokens) > max_seq_length:
            subtokens = ['[CLS]'] + subtokens[-max_seq_length + 1:]
            all_input_mask[i] = [1] + all_input_mask[i][-max_seq_length + 1:]
            all_loss_mask[i] = [not ignore_start_end] + \
                all_loss_mask[i][-max_seq_length + 1:]
            all_subtokens_mask[i] = [False] + \
                all_subtokens_mask[i][-max_seq_length + 1:]

            if with_label:
                all_slots[i] = [pad_id] + all_slots[i][-max_seq_length + 1:]
            too_long_count += 1

        all_input_ids.append([tokenizer.tokens_to_ids(t)
                              for t in subtokens])

        if len(subtokens) < max_seq_length:
            extra = (max_seq_length - len(subtokens))
            all_input_ids[i] = all_input_ids[i] + [0] * extra
            all_loss_mask[i] = all_loss_mask[i] + [False] * extra
            all_subtokens_mask[i] = all_subtokens_mask[i] + [False] * extra
            all_input_mask[i] = all_input_mask[i] + [0] * extra

            if with_label:
                all_slots[i] = all_slots[i] + [pad_id] * extra

        all_segment_ids.append([0] * max_seq_length)

    logger.info(f'{too_long_count} are longer than {max_seq_length}')
    
    for i in range(min(len(all_input_ids), 5)):
        logger.info("*** Example ***")
        logger.info("i: %s" % (i))
        logger.info(
            "subtokens: %s" % " ".join(list(map(str, all_subtokens[i]))))
        logger.info(
            "loss_mask: %s" % " ".join(list(map(str, all_loss_mask[i]))))
        logger.info(
            "input_mask: %s" % " ".join(list(map(str, all_input_mask[i]))))
        logger.info(
            "subtokens_mask: %s" % " ".join(list(map(str, all_subtokens_mask[i]))))
        if with_label:
            logger.info(
                "slots: %s" % " ".join(list(map(str, all_slots[i]))))

    return (all_input_ids,
            all_segment_ids,
            all_input_mask,
            all_loss_mask,
            all_subtokens_mask,
            all_slots,
            label_ids)

class BertTokenClassificationDataset(Dataset):
    """
    Creates dataset to use during training for a token classification
    tasks with pretrained model.

    Converts from raw data to an instance that can be used by
    NMDataLayer.

    For dataset to use during inference without labels, see
    BertTokenClassificationInferDataset.

    Args:
        text_file (str): file to sequences, each line should a sentence,
            No header.
        label_file (str): file to labels, each line corresponds to
            word labels for a sentence in the text_file. No header.
        max_seq_length (int): max sequence length minus 2 for [CLS] and [SEP]
        tokenizer (Tokenizer): such as NemoBertTokenizer
        num_samples (int): number of samples you want to use for the dataset.
            If -1, use all dataset. Useful for testing.
        shuffle (bool): whether to shuffle your data.
        pad_label (str): pad value use for labels.
            by default, it's the neutral label.

    """

    def __init__(self,
                 text_file,
                 label_file,
                 max_seq_length,
                 tokenizer,
                 num_samples=-1,
                 shuffle=False,
                 pad_label='O'):
        if num_samples == 0:
            raise ValueError("num_samples has to be positive", num_samples)
        
        with open(text_file, 'r') as f:
            text_lines = f.readlines()
        
        # Collect all possible labels
        all_labels = set([])
        labels_lines = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip().split()
                labels_lines.append(line)
                all_labels.update(line)

        if len(labels_lines) != len(text_lines):
            raise ValueError("Labels file should contain labels for every word")

        if shuffle or num_samples > 0:
            dataset = list(zip(text_lines, labels_lines))
            random.shuffle(dataset)

            if num_samples > 0:
                dataset = dataset[:num_samples]

            dataset = list(zip(*dataset))
            inputs = dataset[0]
            labels_lines = dataset[1]

        features = get_features(text_lines,
                                max_seq_length,
                                tokenizer,
                                pad_label=pad_label,
                                raw_slots=labels_lines,
                                all_labels=all_labels)
        
        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_loss_mask = features[3]
        self.all_subtokens_mask = features[4]
        self.all_slots = features[5]
        self.label_ids = features[6]
   
        infold = text_file[:text_file.rfind('/')]
        merged_slots = itertools.chain.from_iterable(self.all_slots)
        logger.info('Three most popular slots')
        utils.get_label_stats(merged_slots, infold + '/slot_stats.tsv')

        # save label_ids 
        out = open(infold + '/label_ids.csv', 'w')
        labels, _ = zip(*sorted(self.label_ids.items() ,  key=lambda x: x[1]))
        out.write('\n'.join(labels))
        logger.info(f'Labels: {self.label_ids}')
        logger.info(f'Labels mapping saved to : {out.name}')

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return  (np.array(self.all_input_ids[idx]),
                np.array(self.all_segment_ids[idx]),
                np.array(self.all_input_mask[idx], dtype=np.float32),
                np.array(self.all_loss_mask[idx]),
                np.array(self.all_subtokens_mask[idx]),
                np.array(self.all_slots[idx]))


class BertTokenClassificationInferDataset(Dataset):
    """
    Creates dataset to use during inference for a token classification
    tasks with pretrained model.

    Converts from raw data to an instance that can be used by
    NMDataLayer.

    For dataset to use during training with labels, see
    BertTokenClassificationDataset.

    Args:
        text_file (str): file to sequences, each line should a sentence,
            No header.
        label_file (str): file to labels, each line corresponds to
            word labels for a sentence in the text_file. No header.
        max_seq_length (int): max sequence length minus 2 for [CLS] and [SEP]
        tokenizer (Tokenizer): such as NemoBertTokenizer
        num_samples (int): number of samples you want to use for the dataset.
            If -1, use all dataset. Useful for testing.
        shuffle (bool): whether to shuffle your data.
        pad_label (str): pad value use for labels.
            by default, it's the neutral label.

    """

    def __init__(self,
                 queries,
                 max_seq_length,
                 tokenizer):

        features = get_features(queries,
                                max_seq_length,
                                tokenizer)
        
        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_loss_mask = features[3]
        self.all_subtokens_mask = features[4]

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return  (np.array(self.all_input_ids[idx]),
                np.array(self.all_segment_ids[idx]),
                np.array(self.all_input_mask[idx], dtype=np.float32),
                np.array(self.all_loss_mask[idx]),
                np.array(self.all_subtokens_mask[idx]))
