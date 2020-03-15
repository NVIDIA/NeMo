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

from nemo import logging
from nemo.collections.nlp.data.datasets.datasets_utils import calc_class_weights, get_label_stats, if_exist
from nemo.collections.nlp.utils import get_vocab

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
        none_slot_label (str): the label for slots that aren't identified
            defaulted to 'O'
        pad_label (int): the int used for padding. If set to -1,
             it'll be set to the whatever the None label is.
    """

    def __init__(self, data_dir, none_slot_label='O', pad_label=-1):
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
        slots = JointIntentSlotDataDesc.label2idx(self.slot_dict_file)
        self.num_slots = len(slots)

        for mode in ['train', 'test', 'dev']:
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

    @staticmethod
    def label2idx(file):
        lines = open(file, 'r').readlines()
        lines = [line.strip() for line in lines if line.strip()]
        labels = {lines[i]: i for i in range(len(lines))}
        return labels
