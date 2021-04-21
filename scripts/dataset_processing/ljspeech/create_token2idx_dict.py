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
Creates a dictionary from token to index based on dictionary .txt given.
"""
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dictionary', required=True, default=None, type=str)
parser.add_argument('--dict_out', required=True, default=None, type=str)
args = parser.parse_args()


def main():
    if not os.path.exists(args.dictionary):
        raise FileNotFoundError(f"Could not find dictionary file {args.dictionary}")

    phonemes = set()
    word2phones = {}
    with open(args.dictionary, 'r') as f:
        for line in f:
            line = line.split()
            word = line[0]
            tokens = line[1:]

            word2phones[word] = tokens
            phonemes.update(tokens)

    # Small list of additional punctuation
    word2phones[','] = [' ']
    word2phones[';'] = [' ']
    word2phones['.'] = [' ']
    word2phones['!'] = [' ']
    word2phones['?'] = [' ']
    word2phones['"'] = [' ']
    word2phones['-'] = [' ']

    phone2idx = {k: i for i, k in enumerate(phonemes)}
    phone2idx[' '] = len(phone2idx)
    phone2idx['sil'] = phone2idx[' ']  # Silence
    phone2idx['sp'] = phone2idx[' ']  # Space
    phone2idx['spn'] = phone2idx[' ']  # OOV/unk

    dicts = {
        'phone2idx': phone2idx,
        'word2phones': word2phones,
    }
    with open(args.dict_out, 'w') as f:
        json.dump(dicts, f)

    print(f"Total number of phone indices: {len(phone2idx)}")


if __name__ == '__main__':
    main()
