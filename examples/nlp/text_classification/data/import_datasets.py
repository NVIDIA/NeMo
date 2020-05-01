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

import argparse
import csv
import glob
import json
import os
from os.path import exists

import tqdm

from nemo import logging
from nemo.collections.nlp.data.datasets.datasets_utils.data_preprocessing import DATABASE_EXISTS_TMP, if_exist


def process_imdb(infold, outfold, uncased, modes=['train', 'test']):
    if not os.path.exists(infold):
        link = 'www.kaggle.com/iarunava/imdb-movie-reviews-dataset'
        raise ValueError(f'Data not found at {infold}. ' f'Please download IMDB from {link}.')

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
            files = glob.glob(f'{infold}/{mode}/{sent}/*.txt')
            for file in files:
                with open(file, 'r') as f:
                    review = f.read().strip()
                if uncased:
                    review = review.lower()
                review = review.replace("<br />", "")
                outfiles[mode].write(f'{review}\t{label}\n')
    for mode in modes:
        outfiles[mode].close()


def process_chemprot(source_dir, target_dir, uncased, modes=['train', 'test', 'dev']):
    if not os.path.exists(source_dir):
        link = 'https://github.com/arwhirang/recursive_chemprot/tree/master/Demo/tree_LSTM/data'
        raise ValueError(f'Data not found at {source_dir}. ' f'Please download ChemProt from {link}.')

    logging.info(f'Processing Chemprot dataset and store at {target_dir}')
    os.makedirs(target_dir, exist_ok=True)

    naming_map = {'train': 'trainingPosit_chem', 'test': 'testPosit_chem', 'dev': 'developPosit_chem'}

    def _read_tsv(input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    outfiles = {}
    label_mapping = {}
    out_label_mapping = open(os.path.join(target_dir, 'label_mapping.tsv'), 'w')
    for mode in modes:
        outfiles[mode] = open(os.path.join(target_dir, mode + '.tsv'), 'w')
        outfiles[mode].write('sentence\tlabel\n')
        input_file = os.path.join(source_dir, naming_map[mode])
        lines = _read_tsv(input_file)
        for line in lines:
            text = line[1]
            label = line[2]
            if label == "True":
                label = line[3]
            if uncased:
                text = text.lower()
            if label not in label_mapping:
                out_label_mapping.write(f'{label}\t{len(label_mapping)}\n')
                label_mapping[label] = len(label_mapping)
            label = label_mapping[label]
            outfiles[mode].write(f'{text}\t{label}\n')
    for mode in modes:
        outfiles[mode].close()
    out_label_mapping.close()


def process_thucnews(infold, outfold):
    modes = ['train', 'test']
    train_size = 0.8
    if not os.path.exists(infold):
        link = 'thuctc.thunlp.org/'
        raise ValueError(f'Data not found at {infold}. ' f'Please download THUCNews from {link}.')

    logging.info(f'Processing THUCNews dataset and store at {outfold}')
    os.makedirs(outfold, exist_ok=True)

    outfiles = {}
    for mode in modes:
        outfiles[mode] = open(os.path.join(outfold, mode + '.tsv'), 'a+', encoding='utf-8')
        outfiles[mode].write('sentence\tlabel\n')
    categories = ['体育', '娱乐', '家居', '彩票', '房产', '教育', '时尚', '时政', '星座', '游戏', '社会', '科技', '股票', '财经']
    for category in categories:
        label = categories.index(category)
        category_files = glob.glob(f'{infold}/{category}/*.txt')
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


def process_nlu(filename, outfold, uncased, dataset_name, modes=['train', 'test']):
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


if __name__ == "__main__":
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser(description="Process and convert datasets into NeMo\'s format.")
    parser.add_argument(
        "--dataset_name",
        required=True,
        type=str,
        choices=['sst-2', 'imdb', 'thucnews', 'nlu-chat', 'nlu-ubuntu', 'nlu-web', 'chemprot'],
    )
    parser.add_argument(
        "--source_data_dir", required=True, type=str, help='The path to the folder containing the dataset files.'
    )
    parser.add_argument("--target_data_dir", required=True, type=str)
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument(
        "--ignore_prev_intent",
        action='store_true',
        help='ignores previous intent while importing datasets in jarvis\'s format',
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    do_lower_case = args.do_lower_case
    source_dir = args.source_data_dir
    target_dir = args.target_data_dir

    if not exists(source_dir):
        raise FileNotFoundError(f"{source_dir} does not exist.")

    if dataset_name == 'sst-2':
        logging.info("sst-2 is compatible with NeMo's format and no need for conversion.")
    elif dataset_name == 'imdb':
        process_imdb(source_dir, target_dir, do_lower_case)
    elif dataset_name == 'thucnews':
        process_thucnews(source_dir, target_dir)
    elif dataset_name.startswith('nlu-'):
        if dataset_name == 'nlu-chat':
            infile = f'{source_dir}/ChatbotCorpus.json'
        elif dataset_name == 'nlu-ubuntu':
            infile = f'{source_dir}/AskUbuntuCorpus.json'
        elif dataset_name == 'nlu-web':
            infile = f'{source_dir}/WebApplicationsCorpus.json'
        process_nlu(filename=infile, outfold=target_dir, uncased=do_lower_case, dataset_name=dataset_name)
    elif dataset_name == "chemprot":
        process_chemprot(source_dir, target_dir, do_lower_case)
    else:
        raise ValueError(
            f'Dataset {dataset_name} is not supported.'
            + "Please make sure that you build the preprocessing process for it. "
            + "NeMo's format assumes that a data file has a header and each line of the file follows "
            + "the format: text [TAB] label. Label is assumed to be an integer."
        )
