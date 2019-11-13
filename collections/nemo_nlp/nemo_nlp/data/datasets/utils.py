from collections import Counter
from io import open
import csv
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys

import numpy as np
from sentencepiece import SentencePieceTrainer as SPT
from tqdm import tqdm

from nemo.utils.exp_logging import get_logger

from ...utils.nlp_utils import (get_vocab,
                                write_vocab,
                                write_vocab_in_order,
                                label2idx)


logger = get_logger('')
LOGGING_TMP = '{} dataset has already been processed and stored at {}'


def get_stats(lengths):
    lengths = np.asarray(lengths)
    logger.info(f'Min: {np.min(lengths)} | \
                 Max: {np.max(lengths)} | \
                 Mean: {np.mean(lengths)} | \
                 Median: {np.median(lengths)}')
    logger.info(f'75 percentile: {np.percentile(lengths, 75)}')
    logger.info(f'99 percentile: {np.percentile(lengths, 99)}')


def get_label_stats(labels, outfile='stats.tsv'):
    labels = Counter(labels)
    total = sum(labels.values())
    out = open(outfile, 'w')
    i = 0
    for k, v in labels.most_common():
        out.write(f'{k}\t{v/total}\n')
        if i < 3:
            logger.info(f'{i} item: {k}, {v} out of {total}, {v/total}.')
        i += 1


def list2str(l):
    return ' '.join([str(x) for x in l])


def if_exist(outfold, files):
    if not os.path.exists(outfold):
        return False
    for file in files:
        if not os.path.exists(f'{outfold}/{file}'):
            return False
    return True


def process_sst_2(data_dir):
    if not os.path.exists(data_dir):
        link = 'https://gluebenchmark.com/tasks'
        raise ValueError(f'Data not found at {data_dir}. '
                         f'Please download SST-2 from {link}.')
    logger.info('Keep in mind that SST-2 is only available in lower case.')
    return data_dir


def process_imdb(data_dir, uncased, modes=['train', 'test']):
    if not os.path.exists(data_dir):
        link = 'www.kaggle.com/iarunava/imdb-movie-reviews-dataset'
        raise ValueError(f'Data not found at {data_dir}. '
                         f'Please download IMDB from {link}.')

    outfold = f'{data_dir}/nemo-processed'

    if uncased:
        outfold = f'{outfold}_uncased'

    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        logger.info(LOGGING_TMP.format('IMDB', outfold))
        return outfold
    logger.info(f'Processing IMDB dataset and store at {outfold}')

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

    return outfold


def process_nlu(filename,
                uncased,
                modes=['train', 'test'],
                dataset_name='nlu-ubuntu'):
    """ Dataset has to be of:
    - ubuntu
    - chat
    - web
    """

    if not os.path.exists(filename):
        link = 'https://github.com/sebischair/NLU-Evaluation-Corpora'
        raise ValueError(f'Data not found at {filename}. '
                         'Please download IMDB from {link}.')

    if dataset_name == 'nlu-ubuntu':
        INTENT = {'makeupdate': 1,
                  'setupprinter': 2,
                  'shutdowncomputer': 3,
                  'softwarerecommendation': 4,
                  'none': 0}
    elif dataset_name == 'nlu-chat':
        INTENT = {'departuretime': 0, 'findconnection': 1}
    elif dataset_name == 'nlu-web':
        INTENT = {'changepassword': 1,
                  'deleteaccount': 2,
                  'downloadvideo': 3,
                  'exportdata': 4,
                  'filterspam': 5,
                  'findalternative': 6,
                  'syncaccounts': 7,
                  'none': 0}
    else:
        raise ValueError(f'{dataset_name}: Invalid dataset name')

    infold = filename[:filename.rfind('/')]
    outfold = f'{infold}/{dataset_name}-nemo-processed'

    if uncased:
        outfold = f'{outfold}_uncased'

    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        logger.info(LOGGING_TMP.format(dataset_name.upper(), outfold))
        return outfold
    logger.info(f'Processing data and store at {outfold}')

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
    return outfold


def get_car_labels(intent_file):
    labels = {}
    with open(intent_file, 'r') as f:
        for line in f:
            intent, label = line.strip().split('\t')
            labels[intent] = int(label)
    return labels


def process_nvidia_car(infold,
                       uncased,
                       modes=['train', 'test'],
                       test_ratio=0.02):
    infiles = {'train': f'{infold}/pytextTrainDataPOI_1_0.tsv',
               'test': f'{infold}/test.tsv'}
    outfold = f'{infold}/nvidia-car-nemo-processed'
    intent_file = f'{outfold}/intent_labels.tsv'

    if uncased:
        outfold = f'{outfold}_uncased'

    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        logger.info(LOGGING_TMP.format('NVIDIA-CAR', outfold))
        labels = get_car_labels(intent_file)
        return outfold, labels
    logger.info(f'Processing this dataset and store at {outfold}')

    os.makedirs(outfold, exist_ok=True)

    outfiles = {}

    for mode in modes:
        outfiles[mode] = open(os.path.join(outfold, mode + '.tsv'), 'w')
        outfiles[mode].write('sentence\tlabel\n')
        intents, sentences = [], []
        start_index = 1

        if mode == 'train':
            all_intents = set()
            start_index = 2

        with open(infiles[mode], 'r') as f:
            for line in f:
                intent, _, sentence = line.strip().split('\t')
                if uncased:
                    sentence = sentence.lower()

                if mode == 'train':
                    all_intents.add(intent)
                intents.append(intent)
                sentences.append(' '.join(sentence.split()[start_index:-1]))

        if mode == 'train':
            i = 0
            labels = {}
            intent_out = open(intent_file, 'w')
            for intent in all_intents:
                labels[intent] = i
                logger.info(f'{intent}\t{i}')
                intent_out.write(f'{intent}\t{i}\n')
                i += 1

        seen, repeat = set(), 0
        for intent, sentence in zip(intents, sentences):
            if sentence in seen:
                if mode == 'test':
                    print(sentence)
                repeat += 1
                continue
            text = f'{sentence}\t{labels[intent]}\n'
            outfiles[mode].write(text)
            seen.add(sentence)
        logger.info(f'{repeat} repeated sentences in {mode}')

    return outfold, labels


def process_twitter_airline(filename, uncased, modes=['train', 'test']):
    """ Dataset from Kaggle:
    https://www.kaggle.com/crowdflower/twitter-airline-sentiment
    """
    pass


def ids2text(ids, vocab):
    return ' '.join([vocab[int(id_)] for id_ in ids])


def process_atis(infold, uncased, modes=['train', 'test'], dev_split=0):
    """ MSFT's dataset, processed by Kaggle
    https://www.kaggle.com/siddhadev/atis-dataset-from-ms-cntk
    """
    outfold = f'{infold}/nemo-processed'
    infold = f'{infold}/data/raw_data/ms-cntk-atis'
    vocab = get_vocab(f'{infold}/atis.dict.vocab.csv')

    if uncased:
        outfold = f'{outfold}-uncased'

    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        logger.info(LOGGING_TMP.format('ATIS', outfold))
        return outfold
    logger.info(f'Processing ATIS dataset and store at {outfold}')

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

    shutil.copyfile(f'{infold}/atis.dict.intent.csv',
                    f'{outfold}/dict.intents.csv')
    shutil.copyfile(f'{infold}/atis.dict.slots.csv',
                    f'{outfold}/dict.slots.csv')

    return outfold


def reverse_dict(entity2value):
    value2entity = {}
    for entity in entity2value:
        for value in entity2value[entity]:
            value2entity[value] = entity
    return value2entity


def map_entities(entity2value, entities):
    for key in entities:
        if 'data' in entities[key]:
            if key not in entity2value:
                entity2value[key] = set([])

            values = []
            for value in entities[key]['data']:
                values.append(value['value'])
                values.extend(value['synonyms'])
            entity2value[key] = entity2value[key] | set(values)

    return entity2value


def get_entities(files):
    entity2value = {}
    for file in files:
        with open(file, 'r') as json_file:
            data = json.load(json_file)
            entity2value = map_entities(entity2value, data['entities'])

    value2entity = reverse_dict(entity2value)
    return entity2value, value2entity


def get_data(files, entity2value, value2entity):
    all_data, all_slots, all_intents = [], set(['O']), set()
    for file in files:
        file_data = []
        with open(file, 'r') as json_file:
            data = json.load(json_file)
            for intent in data['intents']:
                all_intents.add(intent)
                utterances = data['intents'][intent]['utterances']
                for utterance in utterances:
                    tokens, slots = [], []
                    for frag in utterance['data']:
                        frag_tokens = frag['text'].strip().split()
                        tokens.extend(frag_tokens)
                        if 'slot_name' not in frag:
                            slot = 'O'
                        else:
                            slot = frag['slot_name']
                            all_slots.add(slot)
                        slots.extend([slot] * len(frag_tokens))
                    file_data.append((tokens, slots, intent))
        all_data.append(file_data)
    return all_data, all_slots, all_intents


def get_dataset(files, dev_split=0.1):
    entity2value, value2entity = get_entities(files)
    data, slots, intents = get_data(files, entity2value, value2entity)
    if len(data) == 1:
        train, dev = partition(data[0], split=dev_split)
    else:
        train, dev = data[0], data[1]
    return train, dev, slots, intents


def partition(data, split=0.1):
    n = len(data)
    n_dev = int(n * split)
    dev_idx = set(random.sample(range(n), n_dev))
    dev, train = [], []

    for i, item in enumerate(data):
        if i in dev_idx:
            dev.append(item)
        else:
            train.append(item)
    return train, dev


def write_data(data, slot_dict, intent_dict, outfold, mode, uncased):
    intent_file = open(f'{outfold}/{mode}.tsv', 'w')
    intent_file.write('sentence\tlabel\n')
    slot_file = open(f'{outfold}/{mode}_slots.tsv', 'w')
    for tokens, slots, intent in data:
        text = ' '.join(tokens)
        if uncased:
            text = text.lower()
        intent_file.write(f'{text}\t{intent_dict[intent]}\n')
        slots = [str(slot_dict[slot]) for slot in slots]
        slot_file.write(' '.join(slots) + '\n')
    intent_file.close()
    slot_file.close()


def create_dataset(train, dev, slots, intents, uncased, outfold):
    os.makedirs(outfold, exist_ok=True)
    if 'O' in slots:
        slots.remove('O')
    slots = sorted(list(slots)) + ['O']
    intents = sorted(list(intents))
    slots = write_vocab(slots, f'{outfold}/dict.slots.csv')
    intents = write_vocab(intents, f'{outfold}/dict.intents.csv')
    write_data(train, slots, intents, outfold, 'train', uncased)
    write_data(dev, slots, intents, outfold, 'test', uncased)


def process_snips(data_dir, uncased, modes=['train', 'test'], dev_split=0.1):
    if not os.path.exists(data_dir):
        link = 'www.github.com/snipsco/spoken-language'
        '-understanding-research-datasets'
        raise ValueError(f'Data not found at {data_dir}. '
                         'Resquest to download the SNIPS dataset from {link}.')

    outfold = f'{data_dir}/nemo-processed'

    if uncased:
        outfold = f'{outfold}-uncased'

    exist = True
    for dataset in ['light', 'speak', 'all']:
        if if_exist(f'{outfold}/{dataset}', [f'{mode}.tsv' for mode in modes]):
            logger.info(LOGGING_TMP.format(
                'SNIPS-' + dataset.upper(), outfold))
        else:
            exist = False
    if exist:
        return outfold

    logger.info(f'Processing SNIPS dataset and store at {outfold}')

    os.makedirs(outfold, exist_ok=True)

    speak_dir = 'smart-speaker-en-close-field'
    light_dir = 'smart-lights-en-close-field'

    light_files = [f'{data_dir}/{light_dir}/dataset.json']
    speak_files = [f'{data_dir}/{speak_dir}/training_dataset.json']
    speak_files.append(f'{data_dir}/{speak_dir}/test_dataset.json')

    light_train, light_dev, light_slots, light_intents = get_dataset(
        light_files, dev_split)
    speak_train, speak_dev, speak_slots, speak_intents = get_dataset(
        speak_files)

    create_dataset(light_train, light_dev, light_slots,
                   light_intents, uncased, f'{outfold}/light')
    create_dataset(speak_train, speak_dev, speak_slots,
                   speak_intents, uncased, f'{outfold}/speak')
    create_dataset(light_train + speak_train, light_dev + speak_dev,
                   light_slots | speak_slots, light_intents | speak_intents,
                   uncased, f'{outfold}/all')

    return outfold


def list2str(nums):
    return ' '.join([str(num) for num in nums])


def merge(data_dir, subdirs, dataset_name, modes=['train', 'test']):
    outfold = f'{data_dir}/{dataset_name}'
    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        logger.info(LOGGING_TMP.format('SNIPS-ATIS', outfold))
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
                querytext = ""
                slots = ""
                for segment in query['data']:
                    querytext = ''.join([querytext, segment['text']])
                    if 'alias' in segment:
                        for word in segment['text'].split():
                            slots = ' '.join([
                                    slots,
                                    slot_labels.get(segment['alias'])])
                    else:
                        for word in segment['text'].split():
                            slots = ' '.join([slots, slot_labels.get('O')])
                querytext = f'{querytext.strip()}\t{index}\n'
                intent_queries.append(querytext)
                slots = f'{slots.strip()}\n'
                slot_tags.append(slots)
                logger.warning(f'Confirm slot tag correspondence with input '
                        f'tokens since tokenization within dialogflow may'
                        f' be different')
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


def partition_df_data(intent_queries, slot_tags, split=0.1):
    n = len(intent_queries)
    n_dev = int(n * split)
    dev_idx = set(random.sample(range(n), n_dev))
    dev_intents, dev_slots, train_intents, train_slots = [], [], [], []

    dev_intents.append('sentence\tlabel\n')
    train_intents.append('sentence\tlabel\n')

    for i, item in enumerate(intent_queries):
        if i in dev_idx:
            dev_intents.append(item)
            dev_slots.append(slot_tags[i])
        else:
            train_intents.append(item)
            train_slots.append(slot_tags[i])
    return train_intents, train_slots, dev_intents, dev_slots


def process_dialogflow(
        data_dir,
        uncased,
        modes=['train', 'test'],
        dev_split=0.1):
    if not os.path.exists(data_dir):
        link = 'www.dialogflow.com'
        raise ValueError(f'Data not found at {data_dir}. '
                         'Export your dialogflow data from'
                         '{link} and unzip at {data_dir}.')

    outfold = f'{data_dir}/dialogflow/nemo-processed'

    '''TO DO  - check for nemo-processed directory
    already exists. If exists, skip the entire creation steps below. '''

    os.makedirs(outfold, exist_ok=True)

    files = get_intent_query_files_dialogflow(data_dir)

    slot_labels = get_slots_dialogflow(files)

    intent_queries, intent_names, slot_tags = get_intents_slots_dialogflow(
        files,
        slot_labels)

    train_queries, train_slots, test_queries, test_slots = \
        partition_df_data(intent_queries, slot_tags, split=dev_split)

    write_df_files(train_queries, f'{outfold}/train.tsv')
    write_df_files(train_slots, f'{outfold}/train_slots.tsv')

    write_df_files(test_queries, f'{outfold}/test.tsv')
    write_df_files(test_slots, f'{outfold}/test_slots.tsv')

    write_df_files(slot_labels, f'{outfold}/dict.slots.csv')
    write_df_files(intent_names, f'{outfold}/dict.intents.csv')

    return outfold


def write_df_files(data, outfile):
    with open(f'{outfile}', 'w') as f:
        for item in data:
            item = f'{item.strip()}\n'
            f.write(item)


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

    def __init__(self,
                 data_dir,
                 do_lower_case=False,
                 dataset_name='default',
                 none_slot_label='O',
                 pad_label=-1):
        if dataset_name == 'atis':
            self.data_dir = process_atis(data_dir, do_lower_case)
        elif dataset_name == 'snips-atis':
            self.data_dir, self.pad_label = merge(
                data_dir,
                ['ATIS/nemo-processed-uncased',
                 'snips/nemo-processed-uncased/all'],
                dataset_name)
        elif dataset_name == 'dialogflow':
            self.data_dir = process_dialogflow(data_dir, do_lower_case)

        elif dataset_name in set(['snips-light', 'snips-speak', 'snips-all']):
            self.data_dir = process_snips(data_dir, do_lower_case)
            if dataset_name.endswith('light'):
                self.data_dir = f'{self.data_dir}/light'
            elif dataset_name.endswith('speak'):
                self.data_dir = f'{self.data_dir}/speak'
            elif dataset_name.endswith('all'):
                self.data_dir = f'{self.data_dir}/all'
        else:
            if not if_exist(data_dir, ['dict.intents.csv', 'dict.slots.csv']):
                raise FileNotFoundError(
                    "Make sure that your data follows the standard format "
                    "supported by JointIntentSlotDataset. Your data must "
                    "contain dict.intents.csv and dict.slots.csv.")
            self.data_dir = data_dir

        self.intent_dict_file = self.data_dir + '/dict.intents.csv'
        self.slot_dict_file = self.data_dir + '/dict.slots.csv'
        self.num_intents = len(get_vocab(self.intent_dict_file))
        slots = label2idx(self.slot_dict_file)
        self.num_slots = len(slots)
        if pad_label != -1:
            self.pad_label = pad_label
        else:
            if none_slot_label not in slots:
                raise ValueError(f'none_slot_label {none_slot_label} not '
                                 f'found in {self.slot_dict_file}.')
            self.pad_label = slots[none_slot_label]


class SentenceClassificationDataDesc:
    def __init__(self, dataset_name, data_dir, do_lower_case):
        if dataset_name == 'sst-2':
            self.data_dir = process_sst_2(data_dir)
            self.num_labels = 2
            self.eval_file = self.data_dir + '/dev.tsv'
        elif dataset_name == 'imdb':
            self.num_labels = 2
            self.data_dir = process_imdb(data_dir, do_lower_case)
            self.eval_file = self.data_dir + '/test.tsv'
        elif dataset_name.startswith('nlu-'):
            if dataset_name.endswith('chat'):
                self.data_dir = f'{data_dir}/ChatbotCorpus.json'
                self.num_labels = 2
            elif dataset_name.endswith('ubuntu'):
                self.data_dir = f'{data_dir}/AskUbuntuCorpus.json'
                self.num_labels = 5
            elif dataset_name.endswith('web'):
                data_dir = f'{data_dir}/WebApplicationsCorpus.json'
                self.num_labels = 8
            self.data_dir = process_nlu(data_dir,
                                        do_lower_case,
                                        dataset_name=dataset_name)
            self.eval_file = self.data_dir + '/test.tsv'
        elif dataset_name == 'nvidia-car':
            self.data_dir, labels = process_nvidia_car(data_dir, do_lower_case)
            for intent in labels:
                idx = labels[intent]
                logger.info(f'{intent}: {idx}')
            self.num_labels = len(labels)
            self.eval_file = self.data_dir + '/test.tsv'
        else:
            logger.info("Looks like you passed a dataset name that isn't "
                        "already supported by NeMo. Please make sure that "
                        "you build the preprocessing method for it.")

        self.train_file = self.data_dir + '/train.tsv'


def create_vocab_lm(data_dir, do_lower_case):
    if if_exist(data_dir, ['train.txt', 'valid.txt', 'test.txt', 'vocab.txt']):
        logger.info(LOGGING_TMP.format('WikiText', data_dir))
        return data_dir
    logger.info(f'Processing WikiText dataset and store at {data_dir}')

    with open(f'{data_dir}/train.txt', 'r') as file:
        txt = file.read()
        if do_lower_case:
            txt = txt.lower()
        lines = re.split(r'[\n]', txt)
        sentences = [line.strip().split() for line in lines if line.strip()]

        vocab = {"[PAD]": 0, "[SEP]": 1, "[CLS]": 2, "[MASK]": 3}
        idx = 4
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1

    with open(f'{data_dir}/vocab.txt', 'w') as f:
        for word in vocab.keys():
            f.write(word + '\n')
    logger.info(f"Created vocabulary of size {len(vocab)}")
    return data_dir


def download_wkt2(data_dir):
    os.makedirs('data/lm', exist_ok=True)
    logger.warning(f'Data not found at {data_dir}. '
                   f'Download {dataset_name} to data/lm')
    data_dir = 'data/lm/wikitext-2'
    subprocess.call('scripts/get_wkt2.sh')
    return data_dir


class LanguageModelDataDesc:
    def __init__(self, dataset_name, data_dir, do_lower_case):
        if dataset_name == 'wikitext-2':
            if not os.path.exists(data_dir):
                data_dir = download_wkt2(data_dir)
            self.data_dir = create_vocab_lm(data_dir, do_lower_case)
        else:
            logger.info("Looks like you passed a dataset name that isn't "
                        "already supported by NeMo. Please make sure that "
                        "you build the preprocessing method for it.")


def create_vocab_mlm(data_dir,
                     vocab_size,
                     sample_size,
                     special_tokens=['PAD', '[UNK]',
                                     '[CLS]', '[SEP]', '[MASK]'],
                     train_file=''):
    vocab = special_tokens[:]
    bert_dir = f'{data_dir}/bert'
    if if_exist(bert_dir, ['tokenizer.model']):
        logger.info(LOGGING_TMP.format('WikiText_BERT', bert_dir))
        return data_dir, f'{bert_dir}/tokenizer.model'
    logger.info(f'Processing WikiText dataset and store at {bert_dir}')
    os.makedirs(bert_dir, exist_ok=True)

    if not train_file:
        files = glob.glob(f'{data_dir}/*.txt')
        train_file = f'{bert_dir}/merged.txt'
        logger.info(f"Merging {len(files)} txt files into {train_file}")

        with open(train_file, "w") as merged:
            for file in tqdm(files):
                with open(file, 'r') as inf:
                    content = inf.read().strip()
                merged.write(content + '\n\n\n')
    else:
        train_file = f'{data_dir}/{train_file}'

    cmd = (f"--input={train_file} --model_prefix={bert_dir}/tokenizer "
           f"--vocab_size={vocab_size - len(vocab)} "
           f"--input_sentence_size={sample_size} "
           f"--shuffle_input_sentence=true --hard_vocab_limit=false "
           f"--bos_id=-1 --eos_id=-1")
    SPT.Train(cmd)

    # Add BERT control symbols
    tokens = []

    with open(f"{bert_dir}/tokenizer.vocab", "r") as f:
        f.readline()  # skip first <unk> token

        # Read tokens from each line and parse for vocab
        for line in f:
            piece = line.split("\t")[0]
            token = piece[1:] if piece.startswith("▁") else f"##{piece}"
            tokens.append(token)

    vocab.extend(tokens)

    # Save vocabulary to output file
    with open(f'{bert_dir}/vocab.txt', "w") as f:
        for token in vocab:
            f.write(f"{token}\n".format())
    return data_dir, f'{bert_dir}/tokenizer.model'


class BERTPretrainingDataDesc:
    def __init__(self,
                 dataset_name,
                 data_dir,
                 vocab_size,
                 sample_size,
                 special_tokens,
                 train_file=''):
        if dataset_name == 'wikitext-2':
            if not os.path.exists(data_dir):
                data_dir = download_wkt2(data_dir)
            self.data_dir, self.tokenizer_model = create_vocab_mlm(
                data_dir,
                vocab_size,
                sample_size,
                special_tokens,
                train_file)
        else:
            logger.info("Looks like you passed a dataset name that isn't "
                        "already supported by NeMo. Please make sure that "
                        "you build the preprocessing method for it.")

        self.train_file = f'{data_dir}/train.txt'
        self.eval_file = f'{data_dir}/valid.txt'
        self.test_file = f'{data_dir}/test.txt'


"""
Utility functions for GLUE tasks
This code was adapted from the HuggingFace library at
https://github.com/huggingface/transformers
"""


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence.
            For single sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second
            sequence. Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


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
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info(f'LOOKING AT {os.path.join(data_dir, "train.tsv")}')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(InputExample(guid=guid,
                                         text_a=text_a,
                                         text_b=text_b,
                                         label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(InputExample(guid=guid,
                                         text_a=text_a,
                                         text_b=text_b,
                                         label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(InputExample(guid=guid,
                                         text_a=text_a,
                                         text_b=None,
                                         label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid,
                                         text_a=text_a,
                                         text_b=None,
                                         label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(InputExample(guid=guid,
                                         text_a=text_a,
                                         text_b=text_b,
                                         label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid,
                                         text_a=text_a,
                                         text_b=text_b,
                                         label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid,
                                         text_a=text_a,
                                         text_b=text_b,
                                         label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid,
                                         text_a=text_a,
                                         text_b=text_b,
                                         label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid,
                                         text_a=text_a,
                                         text_b=text_b,
                                         label=label))
        return examples


processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}
