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
#
# USAGE: python get_conll_data.py --data_dir=/PATH/TO/WHERE/TO/SAVE/DATA

import argparse
import os
import urllib.request

URL = 'https://raw.githubusercontent.com/kyzhouhzau/BERT-NER/master/data/'
URL = {'train.txt': URL + 'train.txt',
       'dev.txt': URL + 'dev.txt'}


def __maybe_download_file(destination: str, source: str):
    """
    Downloads source to destination if not exists.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource
    Returns:
    """
    source = URL[source]
    if not os.path.exists(destination):
        print(f'Downloading {source}')
        urllib.request.urlretrieve(source, filename=destination)


def __process_data(in_file, out_text, out_labels):
    """
    The dataset is splitted into 2 files: text.txt and labels.txt.
    Each line of the text.txt file contains text sequences, where words
    are separated with spaces. The labels.txt file contains corresponding
    labels for each word in text.txt, the labels are separated with spaces.
    Each line of the files should follow the format:
    [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and
    [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).

    """
    in_file = open(in_file, 'r')
    with open(out_text, 'w') as text, open(out_labels, 'w') as labels:
        for line in in_file:
            if line == '\n':
                text.write(line)
                labels.write(line)
            else:
                line = line.split()
                text.write(line[0] + ' ')
                labels.write(line[-1] + ' ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CoNLL-2003 Data download')
    parser.add_argument("--data_dir", required=True, type=str)
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    for dataset in ['dev.txt', 'train.txt']:
        print(f'\nWorking on: {dataset} dataset')
        file_path = os.path.join(args.data_dir, dataset)
        __maybe_download_file(file_path, dataset)

        print(f'Processing {dataset}')
        out_text = os.path.join(args.data_dir, 'text_' + dataset)
        out_labels = os.path.join(args.data_dir, 'labels_' + dataset)
        __process_data(file_path, out_text, out_labels)
        print(f'Processing of the {dataset} is complete')
