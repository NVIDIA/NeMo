#!/usr/bin/env python
# =============================================================================
# Copyright 2019 NVIDIA Corporation. All Rights Reserved.
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

import os
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import re


def create_vocab(lines,
                 vocab_file,
                 min_frequency=3,
                 special_symbols=("[PAD]", "[SEP]", "[CLS]",
                                  "[MASK]", "[UNK]")):
    """Create vocabulary from lines"""
    # Count word occurency
    vocab = {}
    for line in lines:
        if line.strip():
            for w in line.strip().split():
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1

    # Remove rare words
    for w in vocab.keys():
        if vocab[w] < min_frequency:
            del vocab[w]
            print("Remove rare word {} from vocab".format(w))

    # Include special symbols and write to file
    vocab = list(special_symbols) + sorted(vocab.keys())
    with open(vocab_file, 'w') as f:
        for w in vocab:
            f.write(w + '\n')
    print(f"Created vocabulary of size {len(vocab)}")

    return len(vocab)


def read(filename, regex):
    """Read file, filter by regex, add space between characters
    """
    with open(filename, encoding='utf-8') as f:
        documents = f.readlines()

    r = re.compile(regex)
    # Each line is a full article
    for i, line in enumerate(documents):
        text = json.loads(line)['text']
        text = text.replace('\\n', '\n').strip()
        text = r.findall(text)
        text = ' '.join([c for c in text])  # separate char by space
        documents[i] = text

    return documents


def read_files(data_dir, regex, max_dirs=-1):
    """Read text in all sub-directories, filter by regex and return lines"""
    executor = ProcessPoolExecutor(max_workers=4)

    tasks = []
    sub_dirs = [name for name in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, name))]
    for sub_dir in sub_dirs[:max_dirs]:
        p_dir = os.path.join(data_dir, sub_dir)
        print("To process files in dir: "+p_dir)
        files = os.listdir(p_dir)
        for f in files:
            fullpath = os.path.join(data_dir, sub_dir, f)
            tasks.append(executor.submit(partial(read, fullpath, regex)))
    print('Preprocessing texts, please wait...', flush=True)

    # Collect text
    lines = []
    for t in tasks:
        for doc in t.result():
            splitted = re.split(r'[\n]', doc)
            lines += splitted

    return lines


def save(output_dir, lines, train_p=0.95):
    """Save lines into two files within output dir: train.txt and val.txt"""
    # Take large portino as training set
    train_file = os.path.join(output_dir, 'train.txt')
    num_train_doc = int(train_p * len(lines))
    with open(train_file, 'w', encoding='utf-8') as f:
        for l in lines[:num_train_doc]:
            f.write(l+'\n')

    # Take remainings as validation set
    val_file = os.path.join(output_dir, 'valid.txt')
    with open(val_file, 'w', encoding='utf-8') as f:
        for l in lines[num_train_doc:]:
            f.write(l+'\n')

    print("Finished processing.")
    print("Processed text saved to {} and {}".format(train_file, val_file))


def process(data_dir, output_dir=None, min_frequency=3, max_dirs=-1):
    # Define filter rule
    regex = []
    regex += ['[a-zA-Z0-9]']        # English and numerics
    regex += [r'[\u4e00-\u9fff]']   # CJK char
    regex += [r'[\u3400-\u4DBF]']   # CJK char extend
    regex += [r'[\uf900-\ufaff]']   # CJK compatable
    regex += ['[\n]']
    regex = "|".join(regex)

    lines = read_files(data_dir, regex, max_dirs)

    # Create Vocab
    vocab_file = os.path.join(output_dir, "vocab.txt")
    vocab_size = create_vocab(lines, vocab_file, min_frequency)

    output_dir = data_dir if not output_dir
    save(output_dir, lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Process wiki_zh dataset for BERT pretraining')
    # Read data directory from command line argument
    parser.add_argument("--data_dir", default="/raid/data/wiki_zh", type=str)
    parser.add_argument("--output_dir", default="./", type=str)
    parser.add_argument("--min_frequency", default=0, type=int,
                        help="Characters occuring less frequently "
                             "will be filtered out")
    parser.add_argument("--max_dirs", default=-1, type=int,
                        help="Max number of dirs to process")
    args = parser.parse_args()

    process(args.data_dir, args.output_dir, args.min_frequency,
            args.max_dirs)
