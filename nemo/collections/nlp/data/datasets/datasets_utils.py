import csv
import glob
import json
import os
import random
import shutil
import subprocess

from collections import Counter

from collections.nlp.utils.common_nlp_utils import if_exist, get_vocab, list2str, write_vocab_in_order, write_vocab, ids2text
from tqdm import tqdm

import nemo

DATABASE_EXISTS_TMP = '{} dataset has already been processed and stored at {}'
MODE_EXISTS_TMP = '{} mode of {} dataset has already been processed and stored at {}'


def get_label_stats(labels, outfile='stats.tsv'):
    labels = Counter(labels)
    total = sum(labels.values())
    out = open(outfile, 'w')
    i = 0
    label_frequencies = labels.most_common()
    for k, v in label_frequencies:
        out.write(f'{k}\t{v / total}\n')
        if i < 3:
            nemo.logging.info(f'{i} item: {k}, {v} out of {total}, {v / total}.')
        i += 1
    return total, label_frequencies


def process_sst_2(data_dir):
    if not os.path.exists(data_dir):
        link = 'https://gluebenchmark.com/tasks'
        raise ValueError(f'Data not found at {data_dir}. ' f'Please download SST-2 from {link}.')
    nemo.logging.info('Keep in mind that SST-2 is only available in lower case.')
    return data_dir


def process_imdb(data_dir, uncased, modes=['train', 'test']):
    if not os.path.exists(data_dir):
        link = 'www.kaggle.com/iarunava/imdb-movie-reviews-dataset'
        raise ValueError(f'Data not found at {data_dir}. ' f'Please download IMDB from {link}.')

    outfold = f'{data_dir}/nemo-processed'

    if uncased:
        outfold = f'{outfold}_uncased'

    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        nemo.logging.info(DATABASE_EXISTS_TMP.format('IMDB', outfold))
        return outfold
    nemo.logging.info(f'Processing IMDB dataset and store at {outfold}')

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
        nemo.logging.info(DATABASE_EXISTS_TMP.format('THUCNews', outfold))
        return outfold
    nemo.logging.info(f'Processing THUCNews dataset and store at {outfold}')

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
            nemo.logging.info(f'Processing {mode} data of the category {category}')
            if mode == 'test':
                files = test_files
            else:
                files = train_files
            for file in tqdm(files):
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
        raise ValueError(f'Data not found at {filename}. ' 'Please download IMDB from {link}.')

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
        nemo.logging.info(DATABASE_EXISTS_TMP.format(dataset_name.upper(), outfold))
        return outfold
    nemo.logging.info(f'Processing data and store at {outfold}')

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


def process_twitter_airline(filename, uncased, modes=['train', 'test']):
    """ Dataset from Kaggle:
    https://www.kaggle.com/crowdflower/twitter-airline-sentiment
    """
    pass


def process_atis(infold, uncased, modes=['train', 'test'], dev_split=0):
    """ MSFT's dataset, processed by Kaggle
    https://www.kaggle.com/siddhadev/atis-dataset-from-ms-cntk
    """
    outfold = f'{infold}/nemo-processed'
    vocab = get_vocab(f'{infold}/atis.dict.vocab.csv')

    if uncased:
        outfold = f'{outfold}-uncased'

    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        nemo.logging.info(DATABASE_EXISTS_TMP.format('ATIS', outfold))
        return outfold
    nemo.logging.info(f'Processing ATIS dataset and store at {outfold}')

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
        nemo.logging.info(DATABASE_EXISTS_TMP.format(dataset_name, outfold))
        return outfold

    nemo.logging.info(f'Processing {dataset_name} dataset and store at {outfold}')

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
            nemo.logging.info(MODE_EXISTS_TMP.format(mode, dataset_name, outfold, mode))
            continue

        if not if_exist(infold, [f'{mode}.tsv']):
            nemo.logging.info(f'{mode} mode of {dataset_name}' f' is skipped as it was not found.')
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


def process_mturk(data_dir, uncased, modes=['train', 'test'], dev_split=0.1):
    if not os.path.exists(data_dir):
        link = 'www.mturk.com'
        raise ValueError(
            f'Data not found at {data_dir}. ' 'Export your mturk data from' '{link} and unzip at {data_dir}.'
        )

    outfold = f'{data_dir}/nemo-processed'

    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        nemo.logging.info(DATABASE_EXISTS_TMP.format('mturk', outfold))
        return outfold

    nemo.logging.info(f'Processing dataset from mturk and storing at {outfold}')

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
    print(f'agreed_all - {len(agreed_all)}')
    print(f'Slot annotations - {len(slot_annotations)}')

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
        #     print(utterance)

    print(f'inorder utterances - {len(inorder_utterances)}')

    return all_labels, inorder_utterances, slot_tags


def get_intents_mturk(utterances, outfold):
    intent_names = {}
    intent_count = 0

    agreed_all = {}

    print('Printing all intent_labels')
    intent_dict = f'{outfold}/dict.intents.csv'
    if os.path.exists(intent_dict):
        with open(intent_dict, 'r') as f:
            for intent_name in f.readlines():
                intent_names[intent_name.strip()] = intent_count
                intent_count += 1
    print(intent_names)

    for i, utterance in enumerate(utterances[1:]):

        if utterance[1] not in agreed_all:
            agreed_all[utterance[0]] = utterance[1]

        if utterance[1] not in intent_names:
            intent_names[utterance[1]] = intent_count
            intent_count += 1

    print(f'Total number of utterance samples: {len(agreed_all)}')

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


def merge(data_dir, subdirs, dataset_name, modes=['train', 'test']):
    outfold = f'{data_dir}/{dataset_name}'
    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        nemo.logging.info(DATABASE_EXISTS_TMP.format('SNIPS-ATIS', outfold))
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


def partition_data(intent_queries, slot_tags, split=0.1):
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


def write_files(data, outfile):
    with open(outfile, 'w') as f:
        for item in data:
            item = f'{item.strip()}\n'
            f.write(item)


def process_dialogflow(data_dir, uncased, modes=['train', 'test'], dev_split=0.1):
    if not os.path.exists(data_dir):
        link = 'www.dialogflow.com'
        raise ValueError(
            f'Data not found at {data_dir}. ' 'Export your dialogflow data from' '{link} and unzip at {data_dir}.'
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


def read_csv(file_path):
    rows = []
    with open(file_path, 'r') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            rows.append(row)
    return rows


def process_snips(data_dir, uncased, modes=['train', 'test'], dev_split=0.1):
    if not os.path.exists(data_dir):
        link = 'www.github.com/snipsco/spoken-language'
        '-understanding-research-datasets'
        raise ValueError(f'Data not found at {data_dir}. ' 'Resquest to download the SNIPS dataset from {link}.')

    outfold = f'{data_dir}/nemo-processed'

    if uncased:
        outfold = f'{outfold}-uncased'

    exist = True
    for dataset in ['light', 'speak', 'all']:
        if if_exist(f'{outfold}/{dataset}', [f'{mode}.tsv' for mode in modes]):
            nemo.logging.info(DATABASE_EXISTS_TMP.format('SNIPS-' + dataset.upper(), outfold))
        else:
            exist = False
    if exist:
        return outfold

    nemo.logging.info(f'Processing SNIPS dataset and store at {outfold}')

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


def reverse_dict(entity2value):
    value2entity = {}
    for entity in entity2value:
        for value in entity2value[entity]:
            value2entity[value] = entity
    return value2entity


def get_intent_labels(intent_file):
    labels = {}
    label = 0
    with open(intent_file, 'r') as f:
        for line in f:
            intent = line.strip()
            labels[intent] = label
            label += 1
    return labels


def download_wkt2(data_dir):
    os.makedirs('data/lm', exist_ok=True)
    nemo.logging.warning(f'Data not found at {data_dir}. ' f'Downloading wikitext-2 to data/lm')
    data_dir = 'data/lm/wikitext-2'
    subprocess.call('scripts/get_wkt2.sh')
    return data_dir
