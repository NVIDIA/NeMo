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

from nemo import logging
from nemo.collections.nlp.data.datasets.datasets_utils.data_preprocessing import (
    DATABASE_EXISTS_TMP,
    if_exist,
    partition_data,
    read_csv,
    write_files,
)

__all__ = ['process_mturk', 'process_intent_slot_mturk', 'get_intents_mturk', 'get_slot_labels']


def process_mturk(data_dir, uncased, modes=['train', 'test'], dev_split=0.1):
    if not os.path.exists(data_dir):
        link = 'www.mturk.com'
        raise ValueError(
            f'Data not found at {data_dir}. ' f'Export your mturk data from' f'{link} and unzip at {data_dir}.'
        )

    outfold = f'{data_dir}/nemo-processed'

    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        logging.info(DATABASE_EXISTS_TMP.format('mturk', outfold))
        return outfold

    logging.info(f'Processing dataset from mturk and storing at {outfold}')

    os.makedirs(outfold, exist_ok=True)

    classification_data_file = f'{data_dir}/classification.csv'
    annotation_data_file = f'{data_dir}/annotation.manifest'

    if not os.path.exists(classification_data_file):
        raise FileNotFoundError(f'File not found ' f'at {classification_data_file}')

    if not os.path.exists(annotation_data_file):
        raise FileNotFoundError(f'File not found at {annotation_data_file}')

    utterances = []
    utterances = read_csv(classification_data_file)

    # This function assumes that the intent classification data has been
    # reviewed and cleaned and only one label per utterance is present.
    agreed_all, intent_names = get_intents_mturk(utterances, outfold)

    with open(annotation_data_file, 'r') as f:
        slot_annotations = f.readlines()

    # This function assumes that the preprocess step would have made
    # the task_name of all the annotations generic
    task_name = 'retail-combined'

    # It is assumed that every utterances will have corresponding
    # slot annotation information
    if len(slot_annotations) < len(agreed_all):
        raise ValueError(f'Every utterance must have corresponding' f'slot annotation information')

    slot_labels, intent_queries, slot_tags = process_intent_slot_mturk(
        slot_annotations, agreed_all, intent_names, task_name
    )

    assert len(slot_tags) == len(intent_queries)

    dev_split = 0.1

    train_queries, train_slots, test_queries, test_slots = partition_data(intent_queries, slot_tags, split=dev_split)

    write_files(train_queries, f'{outfold}/train.tsv')
    write_files(train_slots, f'{outfold}/train_slots.tsv')

    write_files(test_queries, f'{outfold}/test.tsv')
    write_files(test_slots, f'{outfold}/test_slots.tsv')

    write_files(slot_labels, f'{outfold}/dict.slots.csv')
    write_files(intent_names, f'{outfold}/dict.intents.csv')

    return outfold


def process_intent_slot_mturk(slot_annotations, agreed_all, intent_names, task_name):
    slot_tags = []
    inorder_utterances = []
    all_labels = get_slot_labels(slot_annotations, task_name)
    logging.info(f'agreed_all - {len(agreed_all)}')
    logging.info(f'Slot annotations - {len(slot_annotations)}')

    for annotation in slot_annotations[0:]:
        an = json.loads(annotation)
        utterance = an['source']
        if len(utterance) > 2 and utterance.startswith('"') and utterance.endswith('"'):
            utterance = utterance[1:-1]

        if utterance in agreed_all:
            entities = {}
            annotated_entities = an[task_name]['annotations']['entities']
            for i, each_anno in enumerate(annotated_entities):
                entities[int(each_anno['startOffset'])] = i

            lastptr = 0
            slotlist = []
            # sorting annotations by the start offset
            for i in sorted(entities.keys()):
                annotated_entities = an[task_name]['annotations']['entities']
                tags = annotated_entities[entities.get(i)]
                untagged_words = utterance[lastptr : tags['startOffset']]
                for _ in untagged_words.split():
                    slotlist.append(all_labels.get('O'))
                anno_words = utterance[tags['startOffset'] : tags['endOffset']]
                # tagging with the IOB format.
                for j, _ in enumerate(anno_words.split()):
                    if j == 0:
                        b_slot = 'B-' + tags['label']
                        slotlist.append(all_labels.get(b_slot))
                    else:
                        i_slot = 'I-' + tags['label']
                        slotlist.append(all_labels.get(i_slot))
                lastptr = tags['endOffset']

            untagged_words = utterance[lastptr : len(utterance)]
            for _ in untagged_words.split():
                slotlist.append(all_labels.get('O'))

            slotstr = ' '.join(slotlist)
            slotstr = f'{slotstr.strip()}\n'

            slot_tags.append(slotstr)
            intent_num = intent_names.get(agreed_all.get(utterance))
            query_text = f'{utterance.strip()}\t{intent_num}\n'
            inorder_utterances.append(query_text)
        # else:
        #     logging.warning(utterance)

    logging.info(f'inorder utterances - {len(inorder_utterances)}')

    return all_labels, inorder_utterances, slot_tags


def get_intents_mturk(utterances, outfold):
    intent_names = {}
    intent_count = 0

    agreed_all = {}

    logging.info('Printing all intent_labels')
    intent_dict = f'{outfold}/dict.intents.csv'
    if os.path.exists(intent_dict):
        with open(intent_dict, 'r') as f:
            for intent_name in f.readlines():
                intent_names[intent_name.strip()] = intent_count
                intent_count += 1
    logging.info(intent_names)

    for i, utterance in enumerate(utterances[1:]):

        if utterance[1] not in agreed_all:
            agreed_all[utterance[0]] = utterance[1]

        if utterance[1] not in intent_names:
            intent_names[utterance[1]] = intent_count
            intent_count += 1

    logging.info(f'Total number of utterance samples: {len(agreed_all)}')

    return agreed_all, intent_names


def get_slot_labels(slot_annotations, task_name):
    slot_labels = json.loads(slot_annotations[0])

    all_labels = {}
    count = 0
    # Generating labels with the IOB format.
    for label in slot_labels[task_name]['annotations']['labels']:
        b_slot = 'B-' + label['label']
        i_slot = 'I-' + label['label']
        all_labels[b_slot] = str(count)
        count += 1
        all_labels[i_slot] = str(count)
        count += 1
    all_labels['O'] = str(count)

    return all_labels
