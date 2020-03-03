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

import csv
import glob
import json
import os
import shutil

import tqdm

from nemo import logging
from nemo.collections.nlp.data.datasets.datasets_utils.data_preprocessing import (
    DATABASE_EXISTS_TMP,
    MODE_EXISTS_TMP,
    create_dataset,
    get_dataset,
    if_exist,
)
from nemo.collections.nlp.utils import get_vocab

__all__ = [
    'process_atis',
    'process_jarvis_datasets',
    'process_snips',
    'process_sst_2',
    'process_imdb',
    'process_nlu',
    'process_thucnews',
]


def ids2text(ids, vocab):
    return ' '.join([vocab[int(id_)] for id_ in ids])


def process_atis(infold, uncased, modes=['train', 'test'], dev_split=0):
    """ MSFT's dataset, processed by Kaggle
    https://www.kaggle.com/siddhadev/atis-dataset-from-ms-cntk
    """
    outfold = f'{infold}/nemo-processed'
    vocab = get_vocab(f'{infold}/atis.dict.vocab.csv')

    if uncased:
        outfold = f'{outfold}-uncased'

    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        logging.info(DATABASE_EXISTS_TMP.format('ATIS', outfold))
        return outfold
    logging.info(f'Processing ATIS dataset and store at {outfold}')

    os.makedirs(outfold, exist_ok=True)

    outfiles = {}

    for mode in modes:
        outfiles[mode] = open(os.path.join(outfold, mode + '.tsv'), 'w')
        outfiles[mode].write('sentence\tlabel\n')
        outfiles[mode + '_slots'] = open(f'{outfold}/{mode}_slots.tsv', 'w')

        queries = open(f'{infold}/atis.{mode}.query.csv', 'r').readlines()
        intents = open(f'{infold}/atis.{mode}.intent.csv', 'r').readlines()
        slots = open(f'{infold}/atis.{mode}.slots.csv', 'r').readlines()

        for i, query in enumerate(queries):
            sentence = ids2text(query.strip().split()[1:-1], vocab)
            outfiles[mode].write(f'{sentence}\t{intents[i].strip()}\n')
            slot = ' '.join(slots[i].strip().split()[1:-1])
            outfiles[mode + '_slots'].write(slot + '\n')

    shutil.copyfile(f'{infold}/atis.dict.intent.csv', f'{outfold}/dict.intents.csv')
    shutil.copyfile(f'{infold}/atis.dict.slots.csv', f'{outfold}/dict.slots.csv')
    for mode in modes:
        outfiles[mode].close()

    return outfold


def process_jarvis_datasets(infold, uncased, dataset_name, modes=['train', 'test', 'eval'], ignore_prev_intent=False):
    """ process and convert Jarvis datasets into NeMo's BIO format
    """
    outfold = f'{infold}/{dataset_name}-nemo-processed'
    infold = f'{infold}/'

    if uncased:
        outfold = f'{outfold}-uncased'

    if if_exist(outfold, ['dict.intents.csv', 'dict.slots.csv']):
        logging.info(DATABASE_EXISTS_TMP.format(dataset_name, outfold))
        return outfold

    logging.info(f'Processing {dataset_name} dataset and store at {outfold}')

    os.makedirs(outfold, exist_ok=True)

    outfiles = {}
    intents_list = {}
    slots_list = {}
    slots_list_all = {}

    outfiles['dict_intents'] = open(f'{outfold}/dict.intents.csv', 'w')
    outfiles['dict_slots'] = open(f'{outfold}/dict.slots.csv', 'w')

    outfiles['dict_slots'].write('O\n')
    slots_list["O"] = 0
    slots_list_all["O"] = 0

    for mode in modes:
        if if_exist(outfold, [f'{mode}.tsv']):
            logging.info(MODE_EXISTS_TMP.format(mode, dataset_name, outfold, mode))
            continue

        if not if_exist(infold, [f'{mode}.tsv']):
            logging.info(f'{mode} mode of {dataset_name}' f' is skipped as it was not found.')
            continue

        outfiles[mode] = open(os.path.join(outfold, mode + '.tsv'), 'w')
        outfiles[mode].write('sentence\tlabel\n')
        outfiles[mode + '_slots'] = open(f'{outfold}/{mode}_slots.tsv', 'w')

        queries = open(f'{infold}/{mode}.tsv', 'r').readlines()

        for i, query in enumerate(queries):
            line_splits = query.strip().split("\t")
            if len(line_splits) == 3:
                intent_str, slot_tags_str, sentence = line_splits
            else:
                intent_str, sentence = line_splits
                slot_tags_str = ""

            if intent_str not in intents_list:
                intents_list[intent_str] = len(intents_list)
                outfiles['dict_intents'].write(f'{intent_str}\n')

            if ignore_prev_intent:
                start_token = 2
            else:
                start_token = 1
            sentence_cld = " ".join(sentence.strip().split()[start_token:-1])
            outfiles[mode].write(f'{sentence_cld}\t' f'{str(intents_list[intent_str])}\n')

            slot_tags_list = []
            if slot_tags_str.strip():
                slot_tags = slot_tags_str.strip().split(",")
                for st in slot_tags:
                    if not st.strip():
                        continue
                    [start_i, end_i, slot_name] = st.strip().split(":")
                    slot_tags_list.append([int(start_i), int(end_i), slot_name])
                    if slot_name not in slots_list:
                        slots_list[slot_name] = len(slots_list)
                        slots_list_all[f'B-{slot_name}'] = len(slots_list_all)
                        slots_list_all[f'I-{slot_name}'] = len(slots_list_all)
                        outfiles['dict_slots'].write(f'B-{slot_name}\n')
                        outfiles['dict_slots'].write(f'I-{slot_name}\n')

            slot_tags_list.sort(key=lambda x: x[0])
            slots = []
            processed_index = 0
            for tag_start, tag_end, tag_str in slot_tags_list:
                if tag_start > processed_index:
                    words_list = sentence[processed_index:tag_start].strip().split()
                    slots.extend([str(slots_list_all['O'])] * len(words_list))
                words_list = sentence[tag_start:tag_end].strip().split()
                slots.append(str(slots_list_all[f'B-{tag_str}']))
                slots.extend([str(slots_list_all[f'I-{tag_str}'])] * (len(words_list) - 1))
                processed_index = tag_end

            if processed_index < len(sentence):
                words_list = sentence[processed_index:].strip().split()
                slots.extend([str(slots_list_all['O'])] * len(words_list))

            slots = slots[1:-1]
            slot = ' '.join(slots)
            outfiles[mode + '_slots'].write(slot + '\n')

        outfiles[mode + '_slots'].close()
        outfiles[mode].close()

    outfiles['dict_slots'].close()
    outfiles['dict_intents'].close()

    return outfold


def process_snips(data_dir, uncased, modes=['train', 'test'], dev_split=0.1):
    if not os.path.exists(data_dir):
        link = 'www.github.com/snipsco/spoken-language'
        '-understanding-research-datasets'
        raise ValueError(f'Data not found at {data_dir}. ' f'Resquest to download the SNIPS dataset from {link}.')

    outfold = f'{data_dir}/nemo-processed'

    if uncased:
        outfold = f'{outfold}-uncased'

    exist = True
    for dataset in ['light', 'speak', 'all']:
        if if_exist(f'{outfold}/{dataset}', [f'{mode}.tsv' for mode in modes]):
            logging.info(DATABASE_EXISTS_TMP.format('SNIPS-' + dataset.upper(), outfold))
        else:
            exist = False
    if exist:
        return outfold

    logging.info(f'Processing SNIPS dataset and store at {outfold}')

    os.makedirs(outfold, exist_ok=True)

    speak_dir = 'smart-speaker-en-close-field'
    light_dir = 'smart-lights-en-close-field'

    light_files = [f'{data_dir}/{light_dir}/dataset.json']
    speak_files = [f'{data_dir}/{speak_dir}/training_dataset.json']
    speak_files.append(f'{data_dir}/{speak_dir}/test_dataset.json')

    light_train, light_dev, light_slots, light_intents = get_dataset(light_files, dev_split)
    speak_train, speak_dev, speak_slots, speak_intents = get_dataset(speak_files)

    create_dataset(light_train, light_dev, light_slots, light_intents, uncased, f'{outfold}/light')
    create_dataset(speak_train, speak_dev, speak_slots, speak_intents, uncased, f'{outfold}/speak')
    create_dataset(
        light_train + speak_train,
        light_dev + speak_dev,
        light_slots | speak_slots,
        light_intents | speak_intents,
        uncased,
        f'{outfold}/all',
    )

    return outfold


def process_sst_2(data_dir):
    if not os.path.exists(data_dir):
        link = 'https://gluebenchmark.com/tasks'
        raise ValueError(f'Data not found at {data_dir}. ' f'Please download SST-2 from {link}.')
    logging.info('Keep in mind that SST-2 is only available in lower case.')
    return data_dir


def process_imdb(data_dir, uncased, modes=['train', 'test']):
    if not os.path.exists(data_dir):
        link = 'www.kaggle.com/iarunava/imdb-movie-reviews-dataset'
        raise ValueError(f'Data not found at {data_dir}. ' f'Please download IMDB from {link}.')

    outfold = f'{data_dir}/nemo-processed'

    if uncased:
        outfold = f'{outfold}_uncased'

    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        logging.info(DATABASE_EXISTS_TMP.format('IMDB', outfold))
        return outfold
    logging.info(f'Processing IMDB dataset and store at {outfold}')

    os.makedirs(outfold, exist_ok=True)

    outfiles = {}

    for mode in modes:
        outfiles[mode] = open(os.path.join(outfold, mode + '.tsv'), 'w')
        outfiles[mode].write('sentence\tlabel\n')
        for sent in ['neg', 'pos']:
            if sent == 'neg':
                label = 0
            else:
                label = 1
            files = glob.glob(f'{data_dir}/{mode}/{sent}/*.txt')
            for file in files:
                with open(file, 'r') as f:
                    review = f.read().strip()
                if uncased:
                    review = review.lower()
                review = review.replace("<br />", "")
                outfiles[mode].write(f'{review}\t{label}\n')
    for mode in modes:
        outfiles[mode].close()

    return outfold


def process_thucnews(data_dir):
    modes = ['train', 'test']
    train_size = 0.8
    if not os.path.exists(data_dir):
        link = 'thuctc.thunlp.org/'
        raise ValueError(f'Data not found at {data_dir}. ' f'Please download THUCNews from {link}.')

    outfold = f'{data_dir}/nemo-processed-thucnews'

    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        logging.info(DATABASE_EXISTS_TMP.format('THUCNews', outfold))
        return outfold
    logging.info(f'Processing THUCNews dataset and store at {outfold}')

    os.makedirs(outfold, exist_ok=True)

    outfiles = {}

    for mode in modes:
        outfiles[mode] = open(os.path.join(outfold, mode + '.tsv'), 'a+', encoding='utf-8')
        outfiles[mode].write('sentence\tlabel\n')
    categories = ['体育', '娱乐', '家居', '彩票', '房产', '教育', '时尚', '时政', '星座', '游戏', '社会', '科技', '股票', '财经']
    for category in categories:
        label = categories.index(category)
        category_files = glob.glob(f'{data_dir}/{category}/*.txt')
        test_num = int(len(category_files) * (1 - train_size))
        test_files = category_files[:test_num]
        train_files = category_files[test_num:]

        for mode in modes:
            logging.info(f'Processing {mode} data of the category {category}')
            if mode == 'test':
                files = test_files
            else:
                files = train_files

            if len(files) == 0:
                logging.info(f'Skipping category {category} for {mode} mode')
                continue

            for file in tqdm.tqdm(files):
                with open(file, 'r', encoding='utf-8') as f:
                    news = f.read().strip().replace('\r', '')
                    news = news.replace('\n', '').replace('\t', ' ')
                    outfiles[mode].write(f'{news}\t{label}\n')
    for mode in modes:
        outfiles[mode].close()

    return outfold


def process_nlu(filename, uncased, modes=['train', 'test'], dataset_name='nlu-ubuntu'):
    """ Dataset has to be of:
    - ubuntu
    - chat
    - web
    """

    if not os.path.exists(filename):
        link = 'https://github.com/sebischair/NLU-Evaluation-Corpora'
        raise ValueError(f'Data not found at {filename}. ' f'Please download IMDB from {link}.')

    if dataset_name == 'nlu-ubuntu':
        INTENT = {'makeupdate': 1, 'setupprinter': 2, 'shutdowncomputer': 3, 'softwarerecommendation': 4, 'none': 0}
    elif dataset_name == 'nlu-chat':
        INTENT = {'departuretime': 0, 'findconnection': 1}
    elif dataset_name == 'nlu-web':
        INTENT = {
            'changepassword': 1,
            'deleteaccount': 2,
            'downloadvideo': 3,
            'exportdata': 4,
            'filterspam': 5,
            'findalternative': 6,
            'syncaccounts': 7,
            'none': 0,
        }
    else:
        raise ValueError(f'{dataset_name}: Invalid dataset name')

    infold = filename[: filename.rfind('/')]
    outfold = f'{infold}/{dataset_name}-nemo-processed'

    if uncased:
        outfold = f'{outfold}_uncased'

    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        logging.info(DATABASE_EXISTS_TMP.format(dataset_name.upper(), outfold))
        return outfold
    logging.info(f'Processing data and store at {outfold}')

    os.makedirs(outfold, exist_ok=True)

    outfiles = {}

    for mode in modes:
        outfiles[mode] = open(os.path.join(outfold, mode + '.tsv'), 'w')
        outfiles[mode].write('sentence\tlabel\n')

    with open(filename, 'r') as f:
        data = json.load(f)

    for obj in data['sentences']:
        sentence = obj['text'].strip()
        if uncased:
            sentence = sentence.lower()
        intent = obj['intent'].lower().replace(' ', '')
        label = INTENT[intent]
        txt = f'{sentence}\t{label}\n'
        if obj['training']:
            outfiles['train'].write(txt)
        else:
            outfiles['test'].write(txt)
    for mode in modes:
        outfiles[mode].close()
    return outfold


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                # if sys.version_info[0] == 2:
                #     line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
