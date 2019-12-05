# Copyright (C) NVIDIA CORPORATION. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.****

import argparse
import os
import random
import re
import string
import urllib.request

from collections import Counter

URL = {'tatoeba': 'https://downloads.tatoeba.org/exports/sentences.csv'}


def __maybe_download_file(destination: str, source: str):
    """
    Downloads source to destination if not exists.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource
    """
    source = URL[source]
    if not os.path.exists(destination):
        print(f'Downloading {source}')
        urllib.request.urlretrieve(source, filename=destination)


def __process_english_sentences(in_file,
                                out_file,
                                percent_to_cut,
                                num_to_combine,
                                num_samples=-1):
    """
    Extract English sentences from the dataset that
    contrain letters and punctuation marks (,.?).
    Chop and combine sentences.
    Args:
        in_file: local filepath to the dataset
        out_file: local filepath to the clean dataset
        percent_to_cut: Percent of sentences to cut in the middle
            to get examples of incomplete sentences. This could be useful
            since ASR output not always represents a complete sentence
        num_to_combine: Number of sentences to combine into a single example
        num_samples: Number of samples in the final dataset
    """
    if not os.path.exists(in_file):
        raise FileNotFoundError(f'{in_file} not found.')

    in_file = open(in_file, 'r')
    out_file = open(out_file, 'w')
    lines_to_combine = []
    samples_count = 0

    for line in in_file:
        line = line.split('\t')
        # use only English sentences
        if line[1] == 'eng':
            line = line[2].strip()
            if len(line) > 0 and re.match('^[A-Z][a-z.,?\s]+$', line):  # nopep8
                # chop some sentences in the middle
                if percent_to_cut > 0:
                    line = line.split()
                    if random.random() < percent_to_cut:
                        line = line[:len(line)//2]
                    line = ' '.join(line)

                # combine multiple sentences into a single example
                # to make it harder for the model to learn eos punctuation
                if len(lines_to_combine) > num_to_combine:
                    if samples_count == num_samples:
                        return
                    out_file.write(' '.join(lines_to_combine) + '\n')
                    lines_to_combine = []
                    samples_count += 1
                else:
                    lines_to_combine.append(line)

    if len(lines_to_combine) > 0 and samples_count < num_samples:
        out_file.write(' '.join(lines_to_combine) + '\n')


def __split_into_train_dev(in_file,
                           train_file,
                           dev_file,
                           percent_dev):
    """
    Create train and dev split of the dataset.
    Args:
        in_file: local filepath to the dataset
        train_file: local filepath to the train dataset
        dev_file: local filepath to the dev dataset
        percent_dev: Percent of the sentences in the dev set
    Returns:
        num_lines: number of lines in the clean dataset
    """
    if not os.path.exists(in_file):
        raise FileNotFoundError(f'{in_file} not found.')

    lines = open(in_file, 'r').readlines()
    train_file = open(train_file, 'w')
    dev_file = open(dev_file, 'w')

    dev_size = int(len(lines) * percent_dev)
    train_file.write(' '.join(lines[:-dev_size]))
    dev_file.write(' '.join(lines[-dev_size:]))


def remove_punctuation(word):
    """
    Removes all punctuation marks from a word except for '
    that is often a part of word: don't, it's, and so on
    """
    all_punct_marks = string.punctuation.replace("'", '')
    return re.sub('['+all_punct_marks+']', '', word)


def __create_text_and_labels(data_dir,
                             file,
                             punct_marks=',.?'):
    '''
    Create datasets for training and evaluation.
    The data will be splitted into 2 files: text.txt and labels.txt. \
    Each line of the text.txt file contains text sequences, where words\
    are separated with spaces. The labels.txt file contains \
    corresponding labels for each word in text.txt, the labels are \
    separated with spaces. Each line of the files should follow the \
    format:  \
    [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and \
    [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).'
    '''
    f = open(os.path.join(data_dir, file), 'r')
    text_f = open(os.path.join(data_dir, 'text_' + file), 'w')
    labels_f = open(os.path.join(data_dir, 'labels_' + file), 'w')

    for line in f:
        line = line.split()
        text = ''
        labels = ''
        for word in line:
            label = word[-1] if word[-1] in punct_marks else 'O'
            word = remove_punctuation(word)
            if len(word) > 0:
                if word[0].isupper():
                    label += 'U'
                else:
                    label += 'L'

                word = word.lower()
                text += word + ' '
                labels += label + ' '

        text_f.write(text.strip() + '\n')
        labels_f.write(labels.strip() + '\n')

    __print_stats(labels_f.name)


def __delete_file(file_to_del):
    """
    Deletes the file
    Args:
        file_to_del: local filepath to the file to delete
    """
    if os.path.exists(file_to_del):
        os.remove(file_to_del)


def __print_stats(file_path):
    stats = Counter()
    f = open(file_path, 'r')
    for line in f:
        line = line.split()
        for label in line:
            stats[label] += 1

    print(f'\nStats for {file_path}: {stats}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare tatoeba dataset')
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--dataset", default='tatoeba', type=str)
    parser.add_argument("--num_samples", default=-1, type=int)
    parser.add_argument("--percent_to_cut", default=0, type=float,
                        help='Percent of sentences to cut in the middle')
    parser.add_argument("--num_lines_to_combine", default=1, type=int,
                        help='Number of lines to combine into single example')
    parser.add_argument("--percent_dev", default=0.2, type=float,
                        help='Size of the dev set, percent')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    if args.dataset != 'tatoeba':
        raise ValueError("Unsupported dataset.")

    print(f'Downloading tatoeba dataset')
    tatoeba_dataset = os.path.join(args.data_dir, args.dataset + '.csv')
    __maybe_download_file(tatoeba_dataset, args.dataset)

    print(f'Processing English sentences...')
    eng_sentences = os.path.join(args.data_dir, 'eng_sentences.csv')
    __process_english_sentences(tatoeba_dataset,
                                eng_sentences,
                                args.percent_to_cut,
                                args.num_lines_to_combine,
                                args.num_samples)

    train_file = os.path.join(args.data_dir, 'train.txt')
    dev_file = os.path.join(args.data_dir, 'dev.txt')

    print(f'Splitting the {args.dataset} dataset into train and dev sets')
    __split_into_train_dev(eng_sentences,
                           train_file,
                           dev_file,
                           args.percent_dev)

    __create_text_and_labels(args.data_dir, 'train.txt')
    __create_text_and_labels(args.data_dir, 'dev.txt')

    # clean data_dir
    __delete_file(eng_sentences)
    __delete_file(tatoeba_dataset)
    __delete_file(train_file)
    __delete_file(dev_file)
    print(f'Processing of the {args.dataset} is complete')
