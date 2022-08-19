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


def read_data(args):
    preprocess_file_path = args.preprocess_file_path
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
    print("The new directory " + output_path + " is created!")

    # all dataset need to copy "dict.slots.csv"
    copy_file(args.preprocess_file_path, output_path, "dict.slots.csv")

    # only 'assistant' and 'drive_through' dataset need to copy "dict.intents.csv"
    if args.dataset in ['assistant', 'drive_through']:
        copy_file(args.preprocess_file_path, output_path, "dict.intents.csv")
        copy_file(args.preprocess_file_path, output_path, "dict.slots.csv")


def copy_file(source_path, target_path, filename):
    global_parameters = "*"
    filename_with_path = source_path + filename

    if filename_with_path in glob.glob(os.path.join(source_path, global_parameters)):
        if filename_with_path not in glob.glob(os.path.join(target_path, global_parameters)):
            shutil.copy(filename_with_path, target_path)
            print("The file " + filename_with_path + " is copied!")
        else:
            print("{} exists in {}".format(filename_with_path, os.path.join(os.path.split(target_path)[-2:])))


def from_slot_name_to_slot_id(train, train_slots):

    # manual build dic
    dict_slots = []
    dict_slots = ['O', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']

    dict_slots_id = {}
    for idx, d in enumerate(dict_slots):
        dict_slots_id[d] = str(idx)

    train_new = []
    train_slots_new = []

    for idx, sentence in enumerate(train):
        if idx == 0:
            train_new.append(sentence)
        else:
            new_label = []
            for label in train_slots[idx - 1].strip().split():
                new_label.append(dict_slots_id[label])

            train_new.append(sentence + '\t' + '0')  # every intent label set to 0 at this moment
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
                if word == '.' or word == ',' or word == '?' or word == '\'s':
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
    train_new = []
    train_slots_new = []

    if dataset_name == 'assistant':
        label_for_empty_entity = '54'
    else:
        label_for_empty_entity = '0'

    for idx, sentence in enumerate(train):
        if idx == 0:
            train_new.append(sentence)
        else:
            FLAG_with_entity = False
            for _, label in zip(sentence.split()[:-1], train_slots[idx - 1].strip().split()):
                if label != label_for_empty_entity:
                    FLAG_with_entity = True
                    train_new.append(sentence)
                    train_slots_new.append(train_slots[idx - 1])
                    break
                else:
                    continue
                if FLAG_with_entity == True:
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

    print(args)
    if os.path.exists(args.preprocess_file_path):
        print('folder exist')
        train, train_slots, test, test_slots = read_data(args)

        output_path = args.preprocess_file_path + '/with_entity'
        if not os.path.exists(output_path):
            creat_folder_and_copy_files(args, output_path)

        if args.dataset == 'conll_2003':
            train, train_slots = from_slot_name_to_slot_id(train, train_slots)
            test, test_slots = from_slot_name_to_slot_id(test, test_slots)
            # generate a "dict.intents.csv" file with "same"
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
        print('invalid path')
