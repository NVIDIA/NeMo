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

import json
import os

from nemo.collections.nlp.data.datasets.datasets_utils.data_preprocessing import partition_data, write_files

__all__ = [
    'get_intent_query_files_dialogflow',
    'get_intents_slots_dialogflow',
    'get_slots_dialogflow',
    'process_dialogflow',
]


def get_intent_query_files_dialogflow(path):
    fileslist = []
    for root, _, files in os.walk(path):
        for file in files:
            if '_usersays_en.json' in file:
                fileslist.append(os.path.join(root, file))
    return fileslist


def get_intents_slots_dialogflow(files, slot_labels):
    intent_names = []
    intent_queries = []
    slot_tags = []

    for index, file in enumerate(files):
        intent_names.append(os.path.basename(file).split('_usersays')[0])

        with open(file) as json_file:
            intent_data = json.load(json_file)
            for query in intent_data:
                query_text = ""
                slots = ""
                for segment in query['data']:
                    query_text = ''.join([query_text, segment['text']])
                    if 'alias' in segment:
                        for _ in segment['text'].split():
                            slots = ' '.join([slots, slot_labels.get(segment['alias'])])
                    else:
                        for _ in segment['text'].split():
                            slots = ' '.join([slots, slot_labels.get('O')])
                query_text = f'{query_text.strip()}\t{index}\n'
                intent_queries.append(query_text)
                slots = f'{slots.strip()}\n'
                slot_tags.append(slots)
    return intent_queries, intent_names, slot_tags


def get_slots_dialogflow(files):
    slot_labels = {}
    count = 0
    for file in files:
        intent_head_file = ''.join([file.split('_usersays')[0], '.json'])
        with open(intent_head_file) as json_file:
            intent_meta_data = json.load(json_file)
            for params in intent_meta_data['responses'][0]['parameters']:
                if params['name'] not in slot_labels:
                    slot_labels[params['name']] = str(count)
                    count += 1
    slot_labels['O'] = str(count)
    return slot_labels


def process_dialogflow(data_dir, uncased, modes=['train', 'test'], dev_split=0.1):
    if not os.path.exists(data_dir):
        link = 'www.dialogflow.com'
        raise ValueError(
            f'Data not found at {data_dir}. ' f'Export your dialogflow data from' f'{link} and unzip at {data_dir}.'
        )

    outfold = f'{data_dir}/dialogflow/nemo-processed'

    '''TO DO  - check for nemo-processed directory
    already exists. If exists, skip the entire creation steps below. '''

    os.makedirs(outfold, exist_ok=True)

    files = get_intent_query_files_dialogflow(data_dir)

    slot_labels = get_slots_dialogflow(files)

    intent_queries, intent_names, slot_tags = get_intents_slots_dialogflow(files, slot_labels)

    train_queries, train_slots, test_queries, test_slots = partition_data(intent_queries, slot_tags, split=dev_split)

    write_files(train_queries, f'{outfold}/train.tsv')
    write_files(train_slots, f'{outfold}/train_slots.tsv')

    write_files(test_queries, f'{outfold}/test.tsv')
    write_files(test_slots, f'{outfold}/test_slots.tsv')

    write_files(slot_labels, f'{outfold}/dict.slots.csv')
    write_files(intent_names, f'{outfold}/dict.intents.csv')

    return outfold
