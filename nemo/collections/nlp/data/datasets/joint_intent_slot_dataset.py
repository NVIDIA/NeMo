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
from collections.nlp.data.datasets.datasets_utils import (
    get_label_stats,
    merge,
    process_atis,
    process_dialogflow,
    process_jarvis_datasets,
    process_mturk,
    process_snips,
)
from collections.nlp.utils.common_nlp_utils import calc_class_weights, get_vocab, if_exist, label2idx

import numpy as np
from torch.utils.data import Dataset

import nemo
import nemo.collections.nlp.data.datasets.datasets_utils as utils


def get_features(
    queries,
    max_seq_length,
    tokenizer,
    pad_label=128,
    raw_slots=None,
    ignore_extra_tokens=False,
    ignore_start_end=False,
):
    all_subtokens = []
    all_loss_mask = []
    all_subtokens_mask = []
    all_segment_ids = []
    all_input_ids = []
    all_input_mask = []
    sent_lengths = []
    all_slots = []

    with_label = False
    if raw_slots is not None:
        with_label = True

    for i, query in enumerate(queries):
        words = query.strip().split()
        subtokens = ['[CLS]']
        loss_mask = [1 - ignore_start_end]
        subtokens_mask = [0]
        if with_label:
            slots = [pad_label]

        for j, word in enumerate(words):
            word_tokens = tokenizer.tokenize(word)
            subtokens.extend(word_tokens)

            loss_mask.append(1)
            loss_mask.extend([not ignore_extra_tokens] * (len(word_tokens) - 1))

            subtokens_mask.append(1)
            subtokens_mask.extend([0] * (len(word_tokens) - 1))

            if with_label:
                slots.extend([raw_slots[i][j]] * len(word_tokens))

        subtokens.append('[SEP]')
        loss_mask.append(not ignore_start_end)
        subtokens_mask.append(0)
        sent_lengths.append(len(subtokens))
        all_subtokens.append(subtokens)
        all_loss_mask.append(loss_mask)
        all_subtokens_mask.append(subtokens_mask)
        all_input_mask.append([1] * len(subtokens))
        if with_label:
            slots.append(pad_label)
            all_slots.append(slots)

    max_seq_length = min(max_seq_length, max(sent_lengths))
    nemo.logging.info(f'Max length: {max_seq_length}')
    get_stats(sent_lengths)
    too_long_count = 0

    for i, subtokens in enumerate(all_subtokens):
        if len(subtokens) > max_seq_length:
            subtokens = ['[CLS]'] + subtokens[-max_seq_length + 1 :]
            all_input_mask[i] = [1] + all_input_mask[i][-max_seq_length + 1 :]
            all_loss_mask[i] = [1 - ignore_start_end] + all_loss_mask[i][-max_seq_length + 1 :]
            all_subtokens_mask[i] = [0] + all_subtokens_mask[i][-max_seq_length + 1 :]

            if with_label:
                all_slots[i] = [pad_label] + all_slots[i][-max_seq_length + 1 :]
            too_long_count += 1

        all_input_ids.append([tokenizer._convert_token_to_id(t) for t in subtokens])

        if len(subtokens) < max_seq_length:
            extra = max_seq_length - len(subtokens)
            all_input_ids[i] = all_input_ids[i] + [0] * extra
            all_loss_mask[i] = all_loss_mask[i] + [0] * extra
            all_subtokens_mask[i] = all_subtokens_mask[i] + [0] * extra
            all_input_mask[i] = all_input_mask[i] + [0] * extra

            if with_label:
                all_slots[i] = all_slots[i] + [pad_label] * extra

        all_segment_ids.append([0] * max_seq_length)

    nemo.logging.info(f'{too_long_count} are longer than {max_seq_length}')

    return (all_input_ids, all_segment_ids, all_input_mask, all_loss_mask, all_subtokens_mask, all_slots)


class BertJointIntentSlotDataset(Dataset):
    """
    Creates dataset to use for the task of joint intent
    and slot classification with pretrained model.

    Converts from raw data to an instance that can be used by
    NMDataLayer.

    For dataset to use during inference without labels, see
    BertJointIntentSlotInferDataset.

    Args:
        input_file (str): file to sequence + label.
            the first line is header (sentence [tab] label)
            each line should be [sentence][tab][label]
        slot_file (str): file to slot labels, each line corresponding to
            slot labels for a sentence in input_file. No header.
        max_seq_length (int): max sequence length minus 2 for [CLS] and [SEP]
        tokenizer (Tokenizer): such as BertTokenizer
        num_samples (int): number of samples you want to use for the dataset.
            If -1, use all dataset. Useful for testing.
        shuffle (bool): whether to shuffle your data.
        pad_label (int): pad value use for slot labels.
            by default, it's the neutral label.

    """

    def __init__(
        self,
        input_file,
        slot_file,
        max_seq_length,
        tokenizer,
        num_samples=-1,
        shuffle=True,
        pad_label=128,
        ignore_extra_tokens=False,
        ignore_start_end=False,
    ):
        if num_samples == 0:
            raise ValueError("num_samples has to be positive", num_samples)

        with open(slot_file, 'r') as f:
            slot_lines = f.readlines()

        with open(input_file, 'r') as f:
            input_lines = f.readlines()[1:]

        assert len(slot_lines) == len(input_lines)

        dataset = list(zip(slot_lines, input_lines))

        if shuffle or num_samples > 0:
            random.shuffle(dataset)
        if num_samples > 0:
            dataset = dataset[:num_samples]

        raw_slots, queries, raw_intents = [], [], []
        for slot_line, input_line in dataset:
            raw_slots.append([int(slot) for slot in slot_line.strip().split()])
            parts = input_line.strip().split()
            raw_intents.append(int(parts[-1]))
            queries.append(' '.join(parts[:-1]))

        features = get_features(
            queries,
            max_seq_length,
            tokenizer,
            pad_label=pad_label,
            raw_slots=raw_slots,
            ignore_extra_tokens=ignore_extra_tokens,
            ignore_start_end=ignore_start_end,
        )
        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_loss_mask = features[3]
        self.all_subtokens_mask = features[4]
        self.all_slots = features[5]
        self.all_intents = raw_intents

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return (
            np.array(self.all_input_ids[idx]),
            np.array(self.all_segment_ids[idx]),
            np.array(self.all_input_mask[idx], dtype=np.long),
            np.array(self.all_loss_mask[idx]),
            np.array(self.all_subtokens_mask[idx]),
            self.all_intents[idx],
            np.array(self.all_slots[idx]),
        )


class BertJointIntentSlotInferDataset(Dataset):
    """
    Creates dataset to use for the task of joint intent
    and slot classification with pretrained model.

    Converts from raw data to an instance that can be used by
    NMDataLayer.

    This is to be used during inference only.
    For dataset to use during training with labels, see
    BertJointIntentSlotDataset.

    Args:
        queries (list): list of queries to run inference on
        max_seq_length (int): max sequence length minus 2 for [CLS] and [SEP]
        tokenizer (Tokenizer): such as BertTokenizer
        pad_label (int): pad value use for slot labels.
            by default, it's the neutral label.

    """

    def __init__(self, queries, max_seq_length, tokenizer):
        features = get_features(queries, max_seq_length, tokenizer)

        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_loss_mask = features[3]
        self.all_subtokens_mask = features[4]

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return (
            np.array(self.all_input_ids[idx]),
            np.array(self.all_segment_ids[idx]),
            np.array(self.all_input_mask[idx], dtype=np.long),
            np.array(self.all_loss_mask[idx]),
            np.array(self.all_subtokens_mask[idx]),
        )


class JointIntentSlotDataDesc:
    """ Convert the raw data to the standard format supported by
    JointIntentSlotDataset.

    By default, the None label for slots is 'O'.

    JointIntentSlotDataset requires two files:

        input_file: file to sequence + label.
            the first line is header (sentence [tab] label)
            each line should be [sentence][tab][label]

        slot_file: file to slot labels, each line corresponding to
            slot labels for a sentence in input_file. No header.

    To keep the mapping from label index to label consistent during
    training and inferencing, we require the following files:
        dicts.intents.csv: each line is an intent. The first line
            corresponding to the 0 intent label, the second line
            corresponding to the 1 intent label, and so on.

        dicts.slots.csv: each line is a slot. The first line
            corresponding to the 0 slot label, the second line
            corresponding to the 1 slot label, and so on.

    Args:
        data_dir (str): the directory of the dataset
        do_lower_case (bool): whether to set your dataset to lowercase
        dataset_name (str): the name of the dataset. If it's a dataset
            that follows the standard JointIntentSlotDataset format,
            you can set the name as 'default'.
        none_slot_label (str): the label for slots that aren't indentified
            defaulted to 'O'
        pad_label (int): the int used for padding. If set to -1,
             it'll be set to the whatever the None label is.

    """

    def __init__(self, data_dir, do_lower_case=False, dataset_name='default', none_slot_label='O', pad_label=-1):
        if dataset_name == 'atis':
            self.data_dir = process_atis(data_dir, do_lower_case)
        elif dataset_name == 'snips-atis':
            self.data_dir, self.pad_label = merge(
                data_dir, ['ATIS/nemo-processed-uncased', 'snips/nemo-processed-uncased/all'], dataset_name
            )
        elif dataset_name == 'dialogflow':
            self.data_dir = process_dialogflow(data_dir, do_lower_case)
        elif dataset_name == 'mturk-processed':
            self.data_dir = process_mturk(data_dir, do_lower_case)
        elif dataset_name in set(['snips-light', 'snips-speak', 'snips-all']):
            self.data_dir = process_snips(data_dir, do_lower_case)
            if dataset_name.endswith('light'):
                self.data_dir = f'{self.data_dir}/light'
            elif dataset_name.endswith('speak'):
                self.data_dir = f'{self.data_dir}/speak'
            elif dataset_name.endswith('all'):
                self.data_dir = f'{self.data_dir}/all'
        elif dataset_name.startswith('jarvis'):
            self.data_dir = process_jarvis_datasets(
                data_dir, do_lower_case, dataset_name, modes=["train", "test", "eval"], ignore_prev_intent=False
            )
        else:
            if not if_exist(data_dir, ['dict.intents.csv', 'dict.slots.csv']):
                raise FileNotFoundError(
                    "Make sure that your data follows the standard format "
                    "supported by JointIntentSlotDataset. Your data must "
                    "contain dict.intents.csv and dict.slots.csv."
                )
            self.data_dir = data_dir

        self.intent_dict_file = self.data_dir + '/dict.intents.csv'
        self.slot_dict_file = self.data_dir + '/dict.slots.csv'
        self.num_intents = len(get_vocab(self.intent_dict_file))
        slots = label2idx(self.slot_dict_file)
        self.num_slots = len(slots)

        for mode in ['train', 'test', 'eval']:

            if not if_exist(self.data_dir, [f'{mode}.tsv']):
                nemo.logging.info(f' Stats calculation for {mode} mode' f' is skipped as {mode}.tsv was not found.')
                continue

            slot_file = f'{self.data_dir}/{mode}_slots.tsv'
            with open(slot_file, 'r') as f:
                slot_lines = f.readlines()

            input_file = f'{self.data_dir}/{mode}.tsv'
            with open(input_file, 'r') as f:
                input_lines = f.readlines()[1:]  # Skipping headers at index 0

            if len(slot_lines) != len(input_lines):
                raise ValueError(
                    "Make sure that the number of slot lines match the "
                    "number of intent lines. There should be a 1-1 "
                    "correspondence between every slot and intent lines."
                )

            dataset = list(zip(slot_lines, input_lines))

            raw_slots, queries, raw_intents = [], [], []
            for slot_line, input_line in dataset:
                slot_list = [int(slot) for slot in slot_line.strip().split()]
                raw_slots.append(slot_list)
                parts = input_line.strip().split()
                raw_intents.append(int(parts[-1]))
                queries.append(' '.join(parts[:-1]))

            infold = input_file[: input_file.rfind('/')]

            nemo.logging.info(f'Three most popular intents during {mode}ing')
            total_intents, intent_label_freq = get_label_stats(raw_intents, infold + f'/{mode}_intent_stats.tsv')
            merged_slots = itertools.chain.from_iterable(raw_slots)

            nemo.logging.info(f'Three most popular slots during {mode}ing')
            slots_total, slots_label_freq = get_label_stats(merged_slots, infold + f'/{mode}_slot_stats.tsv')

            if mode == 'train':
                self.slot_weights = calc_class_weights(slots_label_freq)
                nemo.logging.info(f'Slot weights are - {self.slot_weights}')

                self.intent_weights = calc_class_weights(intent_label_freq)
                nemo.logging.info(f'Intent weights are - {self.intent_weights}')

            nemo.logging.info(f'Total intents - {total_intents}')
            nemo.logging.info(f'Intent label frequency - {intent_label_freq}')
            nemo.logging.info(f'Total Slots - {slots_total}')
            nemo.logging.info(f'Slots label frequency - {slots_label_freq}')

        if pad_label != -1:
            self.pad_label = pad_label
        else:
            if none_slot_label not in slots:
                raise ValueError(f'none_slot_label {none_slot_label} not ' f'found in {self.slot_dict_file}.')
            self.pad_label = slots[none_slot_label]


def get_stats(lengths):
    lengths = np.asarray(lengths)
    nemo.logging.info(
        f'Min: {np.min(lengths)} | \
                 Max: {np.max(lengths)} | \
                 Mean: {np.mean(lengths)} | \
                 Median: {np.median(lengths)}'
    )
    nemo.logging.info(f'75 percentile: {np.percentile(lengths, 75)}')
    nemo.logging.info(f'99 percentile: {np.percentile(lengths, 99)}')
