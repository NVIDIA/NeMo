# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""
This script can be used to process and import IMDB, ChemProt, SST-2, and THUCnews datasets into NeMo's format.
You may run it as the following:

python import_datasets.py \
        --dataset_name DATASET_NAME \
        --target_data_dir TARGET_PATH \
        --source_data_dir SOURCE_PATH

The dataset should be specified by "DATASET_NAME" which can be from ["sst-2", "chemprot", "imdb", "thucnews"].
It reads the data from "SOURCE_PATH" folder, processes and converts the data into NeMo's format.
Then writes the results into "TARGET_PATH" folder.

"""

import argparse
import csv
import glob
import os
from os.path import exists

import tqdm

from nemo.utils import logging


def process_imdb(infold, outfold, uncased, modes=['train', 'test']):
    if not os.path.exists(infold):
        link = 'https://ai.stanford.edu/~amaas/data/sentiment/'
        raise ValueError(
            f'Data not found at {infold}. '
            f'Please download IMDB reviews dataset from {link} and '
            f'extract it into the folder specified by source_data_dir argument.'
        )

    logging.info(f'Processing IMDB dataset and store at {outfold}')
    os.makedirs(outfold, exist_ok=True)

    outfiles = {}
    for mode in modes:
        outfiles[mode] = open(os.path.join(outfold, mode + '.tsv'), 'w')
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

    class_labels_file = open(os.path.join(outfold, 'label_ids.tsv'), 'w')
    class_labels_file.write('negative\npositive\n')
    class_labels_file.close()


def process_sst2(infold, outfold, uncased, splits=['train', 'dev']):
    """Process sst2 dataset."""
    # "test" split doesn't have labels, so it is skipped
    if not os.path.exists(infold):
        link = 'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip'
        raise ValueError(
            f'Data not found at {infold}. Please download SST-2 dataset from `{link}` and '
            f'extract it into the folder specified by `source_data_dir` argument.'
        )

    logging.info(f'Processing SST-2 dataset')
    os.makedirs(outfold, exist_ok=True)

    def _read_tsv(input_file, quotechar=None):
        """Read a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    for split in splits:
        # Load input file.
        input_file = os.path.join(infold, split + '.tsv')
        lines = _read_tsv(input_file)
        # Create output.
        outfile = open(os.path.join(outfold, split + '.tsv'), 'w')

        # Copy lines, skip the header (line 0).
        for line in lines[1:]:
            text = line[0]
            label = line[1]
            # Lowercase when required.
            if uncased:
                text = text.lower()
            # Write output.
            outfile.write(f'{text}\t{label}\n')
        # Close file.
        outfile.close()

    class_labels_file = open(os.path.join(outfold, 'label_ids.tsv'), 'w')
    class_labels_file.write('negative\npositive\n')
    class_labels_file.close()

    logging.info(f'Result stored at {outfold}')


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


if __name__ == "__main__":
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser(description="Process and convert datasets into NeMo\'s format.")
    parser.add_argument("--dataset_name", required=True, type=str, choices=['imdb', 'thucnews', 'chemprot'])
    parser.add_argument(
        "--source_data_dir", required=True, type=str, help='The path to the folder containing the dataset files.'
    )
    parser.add_argument("--target_data_dir", required=True, type=str)
    parser.add_argument("--do_lower_case", action='store_true')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    do_lower_case = args.do_lower_case
    source_dir = args.source_data_dir
    target_dir = args.target_data_dir

    if not exists(source_dir):
        raise FileNotFoundError(f"{source_dir} does not exist.")

    if dataset_name == 'imdb':
        process_imdb(source_dir, target_dir, do_lower_case)
    elif dataset_name == 'thucnews':
        process_thucnews(source_dir, target_dir)
    elif dataset_name == "chemprot":
        process_chemprot(source_dir, target_dir, do_lower_case)
    elif dataset_name == "sst-2":
        process_sst2(source_dir, target_dir, do_lower_case)
    else:
        raise ValueError(
            f'Dataset {dataset_name} is not supported.'
            + "Please make sure that you build the preprocessing process for it. "
            + "NeMo's format assumes that a data file has a header and each line of the file follows "
            + "the format: text [TAB] label. Label is assumed to be an integer."
        )
