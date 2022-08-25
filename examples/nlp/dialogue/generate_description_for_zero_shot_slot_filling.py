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
import os
from collections import defaultdict


def read_data(preprocess_file_path):
    with open(preprocess_file_path + '/train.tsv') as f:
        train = f.read().strip().split('\n')

    with open(preprocess_file_path + '/train_slots.tsv') as f:
        train_slots = f.read().strip().split('\n')

    with open(preprocess_file_path + '/dict.slots.csv') as f:
        dict_lines = f.read().strip().split('\n')
        dict_slots = {idx: [slot_name] for idx, slot_name in enumerate(dict_lines)}

    return train, train_slots, dict_slots


def merge_b_and_i_slots_class(dict_slots, dataset_name):
    """
    Some datasets (drive_through and conll_2003) contains slots with separate B-xxx and I-xxx
    such as B-LOC and I-LOC.
    This function seeks to map them into a common slot name (e.g. LOC)

    Args: 
        dict_slots: dict with slot_id as key and slot_name as values
        dataset_name: str specifying dataset used
    Returns:
        merge_dict_slots: dict with merged B- and I- slot_ids as keys and slot_name as values
    """
    merge_dict_slots = {}

    if dataset_name == 'drive_through':
        for slot_id, slot_name in dict_slots.items():
            # slot_names follow the pattern O, xxx, xxx, yyy, yyy ...
            # slot_ids follow the pattern  0, 1,     2,     3,     4, ...
            # merged_slot_ids follow this: 0  1,     1,     2,     2, ...
            merged_slot_id = (slot_id + 1) // 2
            merge_dict_slots[merged_slot_id] = slot_name

    elif dataset_name == 'conll_2003':
        acronym_to_full_name = {'LOC': 'location', 'MISC': 'miscellaneous', 'ORG': 'organization', 'PER': 'person'}
        for slot_id, slot_name in dict_slots.items():
            # slot_names follow the pattern O, B-xxx, I-xxx, B-yyy, I-yyy ...
            # slot_ids follow the pattern  0, 1,     2,     3,     4, ...
            # merged_slot_ids follow this: 0  1,     1,     2,     2, ...
            merged_slot_id = (slot_id + 1) // 2
            slot_name_without_b_or_i = slot_name[0].split('-')[-1]
            merge_dict_slots[merged_slot_id] = [acronym_to_full_name[slot_name_without_b_or_i]]

    # assistant preprocessed data doesn't distinguish between b and i slots
    elif dataset_name == "assistant":
        merge_dict_slots = dict_slots
    return merge_dict_slots


def get_key_words(train, train_slots, dataset_name):
    """
    Identifies key words for each class by counting the 
    number of times each word occurs for samples in each class

    Args:
        train: list of str, each of which represents an utterance
        train_slots: list of slot_ids for each utterance
        dataset_name: str
    
    Returns:
        slot_id_to_keywords: slot_id to dict of keywords with 
        key being each key word (lowercased) and value being their count frequency
    """

    slot_id_to_keywords = {slot_id: defaultdict(int) for slot_id in merge_dict_slots}

    label_for_empty_entity = '54' if dataset_name == 'assistant' else '0'

    # first line of train is not an utterance but header info
    for idx, sentence in enumerate(train[1:]):
        for word, slot_id in zip(sentence.split()[:-1], train_slots[idx].split()):
            if slot_id == label_for_empty_entity:
                continue
            if dataset_name == 'assistant':
                merged_slot_id = int(slot_id)
            else:
                # see reason for this in merge_b_and_i_slots_class
                merged_slot_id = (int(slot_id) + 1) // 2
            slot_id_to_keywords[merged_slot_id][word.lower()] += 1
    return slot_id_to_keywords


def get_class_description(slot_id_to_keywords, dataset_name):
    """
    Combine keywords for each class into a list of descriptions
    Args:
        slot_id_to_keywords: slot_id to dict of keywords with 
            key being each key word (lowercased) and value being their count frequency
        dataset_name: str
    Returns:
        description_list: list of str 

    """
    label_for_empty_entity = '54' if dataset_name == 'assistant' else '0'

    description_list = []
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    punc_set = set(punc)

    for slot_id, slots_keywords in slot_id_to_keywords.items():
        if slot_id == int(label_for_empty_entity):
            description_list.append("other\tother")
        else:
            sorted_keywords = sorted(slots_keywords.items(), key=lambda item: item[1], reverse=True)
            one_slot_description = ' '.join(
                list(
                    set(
                        [keyword for keyword, freq in sorted_keywords if freq > 3]
                        + [keyword for keyword, freq in sorted_keywords[:5]]
                    )
                )
            )
            entity_label = merge_dict_slots[slot_id][0].split("-")[-1]
            entity_label = "".join([ch if ch not in punc_set else ' ' for ch in entity_label])
            description_list.append(entity_label + '\t' + one_slot_description)

    return description_list


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
        train, train_slots, dict_slots = read_data(args.preprocess_file_path)

        merge_dict_slots = merge_b_and_i_slots_class(dict_slots, args.dataset)
        slot_id_to_keywords = get_key_words(train, train_slots, args.dataset)
        description_list = get_class_description(slot_id_to_keywords, args.dataset)

        output_file_name = os.path.join(args.preprocess_file_path, 'description.slots.csv')
        with open(output_file_name, "w") as fw:
            fw.write('\n'.join(description_list))
    else:
        raise ValueError('preprocess_file_path folder {} does not exist'.format(args.preprocess_file_path))
