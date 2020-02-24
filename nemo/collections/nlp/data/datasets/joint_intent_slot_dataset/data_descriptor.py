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

import itertools
import os

from nemo import logging
from nemo.collections.nlp.data.datasets.datasets_utils import (
    DATABASE_EXISTS_TMP,
    calc_class_weights,
    get_label_stats,
    if_exist,
    process_atis,
    process_dialogflow,
    process_jarvis_datasets,
    process_mturk,
    process_snips,
)
from nemo.collections.nlp.utils import get_vocab, list2str

__all__ = ['JointIntentSlotDataDesc']


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
            self.data_dir, self.pad_label = self.merge(
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
                logging.info(f' Stats calculation for {mode} mode' f' is skipped as {mode}.tsv was not found.')
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

            logging.info(f'Three most popular intents during {mode}ing')
            total_intents, intent_label_freq = get_label_stats(raw_intents, infold + f'/{mode}_intent_stats.tsv')
            merged_slots = itertools.chain.from_iterable(raw_slots)

            logging.info(f'Three most popular slots during {mode}ing')
            slots_total, slots_label_freq = get_label_stats(merged_slots, infold + f'/{mode}_slot_stats.tsv')

            if mode == 'train':
                self.slot_weights = calc_class_weights(slots_label_freq)
                logging.info(f'Slot weights are - {self.slot_weights}')

                self.intent_weights = calc_class_weights(intent_label_freq)
                logging.info(f'Intent weights are - {self.intent_weights}')

            logging.info(f'Total intents - {total_intents}')
            logging.info(f'Intent label frequency - {intent_label_freq}')
            logging.info(f'Total Slots - {slots_total}')
            logging.info(f'Slots label frequency - {slots_label_freq}')

        if pad_label != -1:
            self.pad_label = pad_label
        else:
            if none_slot_label not in slots:
                raise ValueError(f'none_slot_label {none_slot_label} not ' f'found in {self.slot_dict_file}.')
            self.pad_label = slots[none_slot_label]

    def merge(self, data_dir, subdirs, dataset_name, modes=['train', 'test']):
        outfold = f'{data_dir}/{dataset_name}'
        if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
            logging.info(DATABASE_EXISTS_TMP.format('SNIPS-ATIS', outfold))
            slots = get_vocab(f'{outfold}/dict.slots.csv')
            none_slot = 0
            for key in slots:
                if slots[key] == 'O':
                    none_slot = key
                    break
            return outfold, int(none_slot)

        os.makedirs(outfold, exist_ok=True)

        data_files, slot_files = {}, {}
        for mode in modes:
            data_files[mode] = open(f'{outfold}/{mode}.tsv', 'w')
            data_files[mode].write('sentence\tlabel\n')
            slot_files[mode] = open(f'{outfold}/{mode}_slots.tsv', 'w')

        intents, slots = {}, {}
        intent_shift, slot_shift = 0, 0
        none_intent, none_slot = -1, -1

        for subdir in subdirs:
            curr_intents = get_vocab(f'{data_dir}/{subdir}/dict.intents.csv')
            curr_slots = get_vocab(f'{data_dir}/{subdir}/dict.slots.csv')

            for key in curr_intents:
                if intent_shift > 0 and curr_intents[key] == 'O':
                    continue
                if curr_intents[key] == 'O' and intent_shift == 0:
                    none_intent = int(key)
                intents[int(key) + intent_shift] = curr_intents[key]

            for key in curr_slots:
                if slot_shift > 0 and curr_slots[key] == 'O':
                    continue
                if slot_shift == 0 and curr_slots[key] == 'O':
                    none_slot = int(key)
                slots[int(key) + slot_shift] = curr_slots[key]

            for mode in modes:
                with open(f'{data_dir}/{subdir}/{mode}.tsv', 'r') as f:
                    for line in f.readlines()[1:]:
                        text, label = line.strip().split('\t')
                        label = int(label)
                        if curr_intents[label] == 'O':
                            label = none_intent
                        else:
                            label = label + intent_shift
                        data_files[mode].write(f'{text}\t{label}\n')

                with open(f'{data_dir}/{subdir}/{mode}_slots.tsv', 'r') as f:
                    for line in f.readlines():
                        labels = [int(label) for label in line.strip().split()]
                        shifted_labels = []
                        for label in labels:
                            if curr_slots[label] == 'O':
                                shifted_labels.append(none_slot)
                            else:
                                shifted_labels.append(label + slot_shift)
                        slot_files[mode].write(list2str(shifted_labels) + '\n')

            intent_shift += len(curr_intents)
            slot_shift += len(curr_slots)

        write_vocab_in_order(intents, f'{outfold}/dict.intents.csv')
        write_vocab_in_order(slots, f'{outfold}/dict.slots.csv')
        return outfold, none_slot


def label2idx(file):
    lines = open(file, 'r').readlines()
    lines = [line.strip() for line in lines if line.strip()]
    labels = {lines[i]: i for i in range(len(lines))}
    return labels


def write_vocab_in_order(vocab, outfile):
    with open(outfile, 'w') as f:
        for key in sorted(vocab.keys()):
            f.write(f'{vocab[key]}\n')
