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
import os

from nemo import logging


def __convert_data(in_file, out_text, out_labels):
    """
    in_file should be in the IOB format, see example here:
    https://www.clips.uantwerpen.be/conll2003/ner/.

    After the convertion, the dataset is splitted into 2 files: text.txt
    and labels.txt.
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
    parser = argparse.ArgumentParser(
        description='Convert data from IOB '
        + 'format to the format compatible with '
        + 'nlp/examples/token_classification.py'
    )
    parser.add_argument("--data_dir", required=True, type=str)
    args = parser.parse_args()

    for dataset in ['dev.txt', 'train.txt']:
        file_path = os.path.join(args.data_dir, dataset)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                "{file_path} not found in {args.data_dir}"
                "For NER, CoNLL-2003 dataset"
                "can be obtained at"
                "https://github.com/kyzhouhzau/BERT"
                "-NER/tree/master/data."
            )

        logging.info(f'Processing {dataset}')
        out_text = os.path.join(args.data_dir, 'text_' + dataset)
        out_labels = os.path.join(args.data_dir, 'labels_' + dataset)

        __convert_data(file_path, out_text, out_labels)
        logging.info(f'Processing of the {dataset} is complete')
