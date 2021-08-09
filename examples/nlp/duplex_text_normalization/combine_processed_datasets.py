# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
This script can be used to combine multiple datasets produced by the script google_data_preprocessing.py
into one single dataset. A potential usecase is to combine multiple datasets of different languages into
one single dataset for multilingual training.

USAGE Example:
python combine_processed_datasets.py
        --input_dirs=PATH_TO_ENGLISH_DATASET_FOLDER
        --input_dirs=PATH_TO_RUSSIAN_DATASET_FOLDER
        --output_dir=PATH_TO_COMBINED_DATASET_FOLDER
"""

from argparse import ArgumentParser
from os import mkdir
from os.path import isdir, join

import nemo.collections.nlp.data.text_normalization.constants as constants
from nemo.collections.nlp.data.text_normalization.utils import read_data_file

if __name__ == '__main__':
    parser = ArgumentParser(description='Combine multiple processed datasets (e.g., for multilingual training)')
    parser.add_argument('--input_dirs', action='append', help='Paths to folders of processed datasets', required=True)
    parser.add_argument('--output_dir', type=str, default='combined', help='Path to the output folder')
    args = parser.parse_args()

    # Create the output dir (if not exist)
    if not isdir(args.output_dir):
        mkdir(args.output_dir)

    # Read input datasets and combine them
    train, dev, test = [], [], []
    for split_name in constants.SPLIT_NAMES:
        if split_name == constants.TRAIN:
            cur_data = train
        if split_name == constants.DEV:
            cur_data = dev
        if split_name == constants.TEST:
            cur_data = test
        # Loop through each input directory
        for input_dir in args.input_dirs:
            input_fp = join(input_dir, f'{split_name}.tsv')
            insts = read_data_file(input_fp)
            cur_data.extend(insts)
    print('After combining the datasets:')
    print(f'len(train): {len(train)}')
    print(f'len(dev): {len(dev)}')
    print(f'len(test): {len(test)}')

    # Output
    for split_name in constants.SPLIT_NAMES:
        output_fp = join(args.output_dir, f'{split_name}.tsv')
        with open(output_fp, 'w+') as output_f:
            if split_name == constants.TRAIN:
                cur_data = train
            if split_name == constants.DEV:
                cur_data = dev
            if split_name == constants.TEST:
                cur_data = test
            for inst in cur_data:
                cur_classes, cur_tokens, cur_outputs = inst
                for c, t, o in zip(cur_classes, cur_tokens, cur_outputs):
                    output_f.write(f'{c}\t{t}\t{o}\n')
                output_f.write('<eos>\t<eos>\n')
