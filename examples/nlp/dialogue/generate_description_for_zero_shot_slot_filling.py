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


def read_data(preprocess_file_path):
    with open(preprocess_file_path + '/train.tsv') as f:
        train = f.read().strip().split('\n')

    with open(preprocess_file_path + '/train_slots.tsv') as f:
        train_slots = f.read().strip().split('\n')

    with open(preprocess_file_path + '/test.tsv') as f:
        test = f.read().strip().split('\n')

    with open(preprocess_file_path + '/test_slots.tsv') as f:
        test_slots = f.read().strip().split('\n')

    with open(preprocess_file_path + '/dict.slots.csv') as f:
        dict_slots = {idx: [slot_name] for idx, slot_name in enumerate(f.read().strip().split('\n'))}
    return train, train_slots, test, test_slots, dict_slots


def merge_b_and_i_slots_class(dict_slots, dataset_name):
    merge_dict_slots = {}
    all_slots_keywords = {}
    merge_dict_slots = {}
    full_name = {'LOC': 'location', 'MISC': 'miscellaneous', 'ORG': 'organization', 'PER': 'person'}

    if dataset_name in ['drive_through', 'conll_2003']:
        for k, v in dict_slots.items():
            if k == 0:
                merge_dict_slots[k] = v
                all_slots_keywords[k] = {}
            else:
                new_id = int((k - 1) / 2) + 1
                merge_dict_slots[new_id] = full_name[v[0].split('-')[1]]
                if dataset_name == 'conll_2003':
                    merge_dict_slots[new_id] = [full_name[v[0].split('-')[1]]]
                else:
                    merge_dict_slots[new_id] = v
                all_slots_keywords[new_id] = {}
    else:
        for k, v in dict_slots.items():
            all_slots_keywords[k] = {}
        merge_dict_slots = dict_slots
    return merge_dict_slots, all_slots_keywords


def get_key_words(train, train_slots, all_slots_keywords, dataset_name):
    if dataset_name == 'assistant':
        label_for_empty_entity = '54'
    else:
        label_for_empty_entity = '0'
    # all_slots_keywords #key: slot_it, value: words
    for idx, sentence in enumerate(train):
        if idx > 0:
            for word, label in zip(sentence.split()[:-1], train_slots[idx - 1].split()):
                if label != label_for_empty_entity:
                    if dataset_name == 'assistant':
                        new_label = int(label)
                    else:
                        new_label = int((int(label) - 1) / 2) + 1
                    new_word = word.lower()
                    if new_word not in all_slots_keywords[new_label]:
                        all_slots_keywords[new_label][new_word] = 1
                    else:
                        all_slots_keywords[new_label][new_word] += 1
    return all_slots_keywords


def get_class_description(all_slots_keywords, dataset_name):
    if dataset_name == 'assistant':
        label_for_empty_entity = '54'
    else:
        label_for_empty_entity = '0'

    description_list = []
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    for k, slots_keywords in all_slots_keywords.items():
        #     print(merge_dict_slots[k])
        if k == int(label_for_empty_entity):
            description_list.append("other\tother")
        else:
            sorted_keywords = sorted(slots_keywords.items(), key=lambda item: item[1], reverse=True)
            one_slot_description = ' '.join(
                list(set([k for k, v in sorted_keywords if v > 3] + [k for k, v in sorted_keywords[:5]]))
            )
            entity_label = merge_dict_slots[k][0].split("-")[-1]
            for c in entity_label:
                if c in punc:
                    entity_label = entity_label.replace(c, " ")

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

    print(args)
    if os.path.exists(args.preprocess_file_path):
        print('folder exist')
        train, train_slots, test, test_slots, dict_slots = read_data(args.preprocess_file_path)
        print(dict_slots)

        merge_dict_slots, all_slots_keywords = merge_b_and_i_slots_class(dict_slots, args.dataset)
        print(merge_dict_slots)
        print(all_slots_keywords)

        all_slots_keywords = get_key_words(train, train_slots, all_slots_keywords, args.dataset)
        description_list = get_class_description(all_slots_keywords, args.dataset)
        print(len(description_list))
        print(description_list)

        output_file_name = os.path.join(args.preprocess_file_path, 'description.slots.csv')

        print(output_file_name)
        with open(output_file_name, "w") as fw:
            fw.write('\n'.join(description_list))
    else:
        raise ValueError("invalid path")
