# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import argparse
import glob
import os
import shutil

from nemo.utils import logging


def read_data(preprocess_file_path):
    if args.dataset == 'conll_2003':
        with open(preprocess_file_path + '/text_train.txt') as f:
            train = f.read().strip().split('\n')
        train = ["sentence\tlabel"] + train
        with open(preprocess_file_path + '/labels_train.txt') as f:
            train_slots = f.read().strip().split('\n')
        with open(preprocess_file_path + '/text_test.txt') as f:
            test = f.read().strip().split('\n')
        test = ["sentence\tlabel"] + test
        with open(preprocess_file_path + '/labels_test.txt') as f:
            test_slots = f.read().strip().split('\n')
    else:
        with open(preprocess_file_path + '/train.tsv') as f:
            train = f.read().strip().split('\n')
        with open(preprocess_file_path + '/train_slots.tsv') as f:
            train_slots = f.read().strip().split('\n')
        with open(preprocess_file_path + '/test.tsv') as f:
            test = f.read().strip().split('\n')
        with open(preprocess_file_path + '/test_slots.tsv') as f:
            test_slots = f.read().strip().split('\n')
    return train, train_slots, test, test_slots


def creat_folder_and_copy_files(args, output_path):
    os.makedirs(output_path)
    logging.info("The new directory " + output_path + " is created!")

    # all dataset need to copy "dict.slots.csv"
    copy_file(args.preprocess_file_path, output_path, "dict.slots.csv")

    # only 'assistant' and 'drive_through' dataset need to copy "dict.intents.csv"
    if args.dataset in ['assistant', 'drive_through']:
        copy_file(args.preprocess_file_path, output_path, "dict.intents.csv")
        copy_file(args.preprocess_file_path, output_path, "dict.slots.csv")


def copy_file(source_path, target_path, filename):
    global_parameters = "*"
    filename_with_path = os.path.join(source_path, filename)

    if filename_with_path in glob.glob(os.path.join(source_path, global_parameters)):
        if filename_with_path not in glob.glob(os.path.join(target_path, global_parameters)):
            shutil.copy(filename_with_path, target_path)
            logging.info("The file " + filename_with_path + " is copied!")
        else:
            logging.info("{} exists in {}".format(filename_with_path, os.path.join(os.path.split(target_path)[-2:])))


def map_conll2003_slot_name_to_slot_id(train, train_slots):

    dict_slots = ['O', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']

    slot_name_to_slot_id = {slot_name: str(idx) for idx, slot_name in enumerate(dict_slots)}

    train_new = []
    train_slots_new = []

    for idx, sentence in enumerate(train):
        if idx == 0:
            train_new.append(sentence)
        else:
            new_label = []
            for label in train_slots[idx - 1].strip().split():
                new_label.append(slot_name_to_slot_id[label])

            # set every intent label set to dummy value of 0
            train_new.append(sentence + '\t' + '0')
            train_slots_new.append(' '.join(new_label))

    return train_new, train_slots_new


def fix_input_data_by_combine_punctuation_with_word(train, train_slots):
    train_new = []
    train_slots_new = []

    for idx, sentence in enumerate(train):
        if idx == 0:
            train_new.append(sentence)
        else:
            new_sentence = []
            new_label = []
            for word, label in zip(sentence.strip().split(), train_slots[idx - 1].strip().split()):
                if word in ['.', ',', '?', '\'s']:
                    if new_sentence:
                        new_sentence[-1] = new_sentence[-1] + word
                else:
                    new_sentence.append(word)
                    new_label.append(label)
            if len(new_sentence) != 0:
                train_new.append(' '.join(new_sentence) + '\t' + sentence.split()[-1])
                train_slots_new.append(' '.join(new_label))

            else:
                train_new.append(sentence)
                train_slots_new.append(train_slots[idx - 1])
    return train_new, train_slots_new


def remove_sentence_slot_label_pair_in_data_if_without_entity(train, train_slots, dataset_name):

    # train[0] is the header and not an utterance
    train_new = [train[0]]
    train_slots_new = []

    label_for_empty_entity = '54' if dataset_name == 'assistant' else '0'

    for idx, sentence in enumerate(train[1:]):
        for label in train_slots[idx].strip().split():
            if label != label_for_empty_entity:
                train_new.append(sentence)
                train_slots_new.append(train_slots[idx])
                break

    return train_new, train_slots_new


def write_file(data, output_path, filename):
    output_filename_with_path = os.path.join(output_path, filename)
    with open(output_filename_with_path, "w") as fw:
        fw.write('\n'.join(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_file_path", help="The folder contains all the dataset ")
    parser.add_argument(
        "--dataset",
        choices=['drive_through', 'assistant', 'conll_2003'],
        default='assistant',
        help="the type of dataset",
    )
    args = parser.parse_args()

    if os.path.exists(args.preprocess_file_path):
        train, train_slots, test, test_slots = read_data(args.preprocess_file_path)

        output_path = args.preprocess_file_path + '/with_entity'
        if not os.path.exists(output_path):
            creat_folder_and_copy_files(args, output_path)

        if args.dataset == 'conll_2003':
            train, train_slots = map_conll2003_slot_name_to_slot_id(train, train_slots)
            test, test_slots = map_conll2003_slot_name_to_slot_id(test, test_slots)
            # generate a "dict.intents.csv" file with "same" as CoNLL2003 does not come with intents
            write_file(['same'], output_path, 'dict.intents.csv')

            # generate a "dict.slots.csv" file with manual build slots class list
            dict_slots = ['O', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']
            write_file(dict_slots, output_path, 'dict.slots.csv')

        if args.dataset in ['conll_2003', 'drive_through']:
            train, train_slots = fix_input_data_by_combine_punctuation_with_word(train, train_slots)
            test, test_slots = fix_input_data_by_combine_punctuation_with_word(test, test_slots)

        train, train_slots = remove_sentence_slot_label_pair_in_data_if_without_entity(
            train, train_slots, args.dataset
        )
        test, test_slots = remove_sentence_slot_label_pair_in_data_if_without_entity(test, test_slots, args.dataset)

        write_file(train, output_path, 'train.tsv')
        write_file(train_slots, output_path, 'train_slots.tsv')
        write_file(test, output_path, 'test.tsv')
        write_file(test_slots, output_path, 'test_slots.tsv')

    else:
        raise ValueError('preprocess_file_path folder {} does not exist'.format(args.preprocess_file_path))
