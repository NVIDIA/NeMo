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
"""
This script downloads and unpacks LibriTTS data. And prepares it for punctuation and capitalization lexical audio model.
Data is being downloaded from www.openslr.org and then extracted via tar.
The script gathers text from every *.normalized.txt file inside of archive into single file with text and file with audio filepaths.
"""
import argparse
import glob
import os
import re
import shutil
import string
import subprocess
import tarfile

from tqdm import tqdm

from nemo.utils import logging

URL = {
    'train_clean_100': "https://www.openslr.org/resources/60/train-clean-100.tar.gz",
    'train_clean_360': "https://www.openslr.org/resources/60/train-clean-360.tar.gz",
    'train_other_500': "https://www.openslr.org/resources/60/train-other-500.tar.gz",
    'dev_clean': "https://www.openslr.org/resources/60/dev-clean.tar.gz",
    'dev_other': "https://www.openslr.org/resources/60/dev-other.tar.gz",
    'test_clean': "https://www.openslr.org/resources/60/test-clean.tar.gz",
    'test_other': "https://www.openslr.org/resources/60/test-other.tar.gz",
}


def remove_punctuation(word: str):
    """
    Removes all punctuation marks from a word except for '
    that is often a part of word: don't, it's, and so on
    """
    all_punct_marks = string.punctuation.replace("'", '')
    return re.sub('[' + all_punct_marks + ']', '', word)


def create_text_and_labels(output_dir: str, file_path: str, punct_marks: str = ',.?'):
    """
    Create datasets for training and evaluation.

    Args:
      output_dir: path to the output data directory
      file_path: path to file name
      punct_marks: supported punctuation marks

    The data will be split into 2 files: text.txt and labels.txt. \
    Each line of the text.txt file contains text sequences, where words\
    are separated with spaces. The labels.txt file contains \
    corresponding labels for each word in text.txt, the labels are \
    separated with spaces. Each line of the files should follow the \
    format:  \
    [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and \
    [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).'
    """
    if not os.path.exists(file_path):
        raise ValueError(f'{file_path} not found')

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(file_path)
    labels_file = os.path.join(output_dir, 'labels_' + base_name)
    text_file = os.path.join(output_dir, 'text_' + base_name)

    with open(file_path, 'r') as f:
        with open(text_file, 'w') as text_f:
            with open(labels_file, 'w') as labels_f:
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
                                label += 'O'

                            word = word.lower()
                            text += word + ' '
                            labels += label + ' '

                    text_f.write(text.strip() + '\n')
                    labels_f.write(labels.strip() + '\n')

    print(f'{text_file} and {labels_file} created from {file_path}.')


def __extract_file(filepath, data_dir):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        print(f"Error while extracting {filepath}. Already extracted?")


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
        logging.info(f'Downloading {source} to {destination}')
        subprocess.run(['wget', '-O', destination, source])
        return 1
    else:
        logging.info(f'{destination} found. Skipping download')
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare LibriTTS dataset for punctuation capitalization lexical audio model training/evaluating.'
    )
    parser.add_argument("--data_sets", default="dev_clean", type=str, help="List of subsets separated by comma")
    parser.add_argument("--data_dir", required=True, type=str, help="Path to dir where data will be stored")
    parser.add_argument(
        "--clean", "-c", action="store_true", help="If set to True will delete all files except produced .txt and .wav"
    )
    args = parser.parse_args()

    data_dir = args.data_dir

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for subset in args.data_sets.split(','):
        logging.info(f'Downloading {subset} subset')
        if __maybe_download_file(data_dir + f'/{subset}.tar.gz', subset):
            logging.info(f'Extracting {subset} subset')
            __extract_file(data_dir + f'/{subset}.tar.gz', data_dir)

    logging.info(f'Processing data')

    splits = set([split.split('_')[0] for split in args.data_sets.split(',')])
    for split in splits:
        os.makedirs(f'{data_dir}/audio/{split}', exist_ok=True)
        with open(f'{data_dir}/{split}.txt', 'w') as text_data, open(
            f'{data_dir}/audio_{split}.txt', 'w'
        ) as audio_data:
            for file in tqdm(glob.glob(f'{data_dir}/LibriTTS/{split}*/*/*/*.wav'), desc=f'Processing {split}'):
                with open(file[:-4] + '.normalized.txt', 'r') as source_file:
                    lines = source_file.readlines()
                    text = lines[0]
                    text = re.sub(r"[^a-zA-Z\d,?!.']", ' ', text)
                    text = re.sub(' +', ' ', text)
                shutil.copy(file.strip(), (f'{data_dir}/audio/{split}/' + file.split('/')[-1]).strip())
                text_data.write(text.strip() + "\n")
                audio_data.write((f'{data_dir}/audio/{split}/' + file.split('/')[-1]).strip() + "\n")
        create_text_and_labels(f'{data_dir}/', f'{data_dir}/{split}.txt')
        logging.info(f'Processed {split} subset')

    if args.clean:
        shutil.rmtree(f'{data_dir}/LibriTTS')
        for tar in glob.glob(f'{data_dir}/**.tar.gz'):
            os.remove(tar)
