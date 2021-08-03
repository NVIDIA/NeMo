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
This script can be used to process the raw data files of the Google Text Normalization dataset
to obtain data files of the format mentioned in the `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`.
For English, the script also does some preprocessing on the spoken forms of the URLs. For example,
given the URL "Zimbio.com", the original expected spoken form in the Google dataset is
"z_letter i_letter m_letter b_letter i_letter o_letter dot c_letter o_letter m_letter".
However, our script will return a more concise output which is "zim bio dot com".


USAGE Example:
1. Download the Google TN dataset from https://www.kaggle.com/google-nlu/text-normalization
2. Unzip the English subset (e.g., by running `tar zxvf  en_with_types.tgz`). Then there will a folder named `en_with_types`.
3. Run this script
# python google_data_preprocessing.py       \
        --data_dir=en_with_types/           \
        --output_dir=preprocessed/          \
        --lang=en

In this example, the final preprocessed files will be stored in the `preprocessed` folder.
The folder should contain three files `train.tsv`, 'dev.tsv', and `test.tsv`. Similar steps
can be used to preprocess the Russian subset.
"""

from argparse import ArgumentParser
from os import listdir, mkdir
from os.path import isdir, isfile, join

import wordninja
from helpers import flatten
from tqdm import tqdm

from nemo.collections.nlp.data.text_normalization import constants
from nemo.collections.nlp.data.text_normalization.utils import basic_tokenize

# Local Constants
MAX_DEV_SIZE = 25000

# Helper Functions
def read_google_data(data_dir, lang):
    """
    The function can be used to read the raw data files of the Google Text Normalization
    dataset (which can be downloaded from https://www.kaggle.com/google-nlu/text-normalization)

    Args:
        data_dir: Path to the data directory. The directory should contain files of the form output-xxxxx-of-00100
        lang: Selected language.
    Return:
        train: A list of examples in the training set.
        dev: A list of examples in the dev set
        test: A list of examples in the test set
    """
    train, dev, test = [], [], []
    for fn in listdir(data_dir):
        fp = join(data_dir, fn)
        if not isfile(fp):
            continue
        if not fn.startswith('output'):
            continue
        with open(fp, 'r', encoding='utf-8') as f:
            # Determine the current split
            split_nb = int(fn.split('-')[1])
            if split_nb < 5:
                # For English, the train data is only from output-00000-of-00100
                # For Russian, the train data is from output-00000-of-00100 to output-00004-of-00100
                if split_nb > 0 and lang == constants.ENGLISH:
                    continue
                cur_split = train
            elif split_nb == 90:
                cur_split = dev
            elif split_nb == 99:
                cur_split = test
            else:
                continue
            # Loop through each line of the file
            cur_classes, cur_tokens, cur_outputs = [], [], []
            for linectx, line in tqdm(enumerate(f)):
                es = line.strip().split('\t')
                if split_nb == 99:
                    # For the results reported in the paper "RNN Approaches to Text Normalization: A Challenge":
                    # + For English, the first 100,002 lines of output-00099-of-00100 are used for the test set
                    # + For Russian, the first 100,007 lines of output-00099-of-00100 are used for the test set
                    if lang == constants.ENGLISH and linectx == 100002:
                        break
                    if lang == constants.RUSSIAN and linectx == 100007:
                        break
                if len(es) == 2 and es[0] == '<eos>':
                    # Update cur_split
                    cur_outputs = process_url(cur_tokens, cur_outputs, lang)
                    cur_split.append((cur_classes, cur_tokens, cur_outputs))
                    # Reset
                    cur_classes, cur_tokens, cur_outputs = [], [], []
                    continue
                # Remove _trans (for Russian)
                if lang == constants.RUSSIAN:
                    es[2] = es[2].replace('_trans', '')
                # Update the current example
                assert len(es) == 3
                cur_classes.append(es[0])
                cur_tokens.append(es[1])
                cur_outputs.append(es[2])
    dev = dev[:MAX_DEV_SIZE]
    train_sz, dev_sz, test_sz = len(train), len(dev), len(test)
    print(f'train_sz: {train_sz} | dev_sz: {dev_sz} | test_sz: {test_sz}')
    return train, dev, test


def process_url(tokens, outputs, lang):
    """
    The function is used to process the spoken form of every URL in an example

    Args:
        tokens: The tokens of the written form
        outputs: The expected outputs for the spoken form
        lang: Selected language.
    Return:
        outputs: The outputs for the spoken form with preprocessed URLs.
    """
    if lang == constants.ENGLISH:
        for i in range(len(tokens)):
            t, o = tokens[i], outputs[i]
            if o != constants.SIL_WORD and '_letter' in o:
                o_tokens = o.split(' ')
                all_spans, cur_span = [], []
                for j in range(len(o_tokens)):
                    if len(o_tokens[j]) == 0:
                        continue
                    if o_tokens[j] == '_letter':
                        all_spans.append(cur_span)
                        all_spans.append([' '])
                        cur_span = []
                    else:
                        o_tokens[j] = o_tokens[j].replace('_letter', '')
                        cur_span.append(o_tokens[j])
                if len(cur_span) > 0:
                    all_spans.append(cur_span)
                o_tokens = flatten(all_spans)

                o = ''
                for o_token in o_tokens:
                    if len(o_token) > 1:
                        o += ' ' + o_token + ' '
                    else:
                        o += o_token
                o = o.strip()
                o_tokens = wordninja.split(o)
                o = ' '.join(o_tokens)

                outputs[i] = o

    return outputs


# Main code
if __name__ == '__main__':
    parser = ArgumentParser(description='Preprocess Google text normalization dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to folder with data')
    parser.add_argument('--output_dir', type=str, default='preprocessed', help='Path to folder with preprocessed data')
    parser.add_argument(
        '--lang', type=str, default=constants.ENGLISH, choices=constants.SUPPORTED_LANGS, help='Language'
    )
    args = parser.parse_args()

    # Create the output dir (if not exist)
    if not isdir(args.output_dir):
        mkdir(args.output_dir)

    # Processing
    train, dev, test = read_google_data(args.data_dir, args.lang)
    for split, data in zip(constants.SPLIT_NAMES, [train, dev, test]):
        output_f = open(join(args.output_dir, f'{split}.tsv'), 'w+', encoding='utf-8')
        for inst in data:
            cur_classes, cur_tokens, cur_outputs = inst
            for c, t, o in zip(cur_classes, cur_tokens, cur_outputs):
                t = ' '.join(basic_tokenize(t, args.lang))
                if not o in constants.SPECIAL_WORDS:
                    o_tokens = basic_tokenize(o, args.lang)
                    o_tokens = [o_tok for o_tok in o_tokens if o_tok != constants.SIL_WORD]
                    o = ' '.join(o_tokens)
                output_f.write(f'{c}\t{t}\t{o}\n')
            output_f.write('<eos>\t<eos>\n')
