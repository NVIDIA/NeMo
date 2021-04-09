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
Calculates durations for LJSpeech based on MFA TextGrid alignments.
"""
import argparse
import glob
import json
from math import ceil
import numpy as np
import os
import re

parser = argparse.ArgumentParser(
    description="Calculates phoneme durations for LJSpeech from TextGrids."
)
parser.add_argument('--ljspeech_dir', required=True, default=None, type=str)
parser.add_argument('--mappings', required=False, default=None, type=str,
    help='JSON file of mappings created with create_token2idx_dict.py')
parser.add_argument('--word_durations', action='store_true',
    help='Calculate word durations instead of phoneme durations')
parser.add_argument('--sr', required=False, default=22050, type=int)
parser.add_argument('--window_stride', required=False, default=256, type=int)
args = parser.parse_args()


def find_nums(s):
    """Extracts numbers of the form x.y and xyz from the given string."""
    return re.findall('[\d]+\.[\d]+|[\d]+', s)


def get_token_and_dur(lines):
    """Given a set of lines (under "intervals") from a TextGrid, gets the token and duration (in secs)."""
    token_line = lines[3]
    token_l, token_r = token_line.find('\"'), token_line.rfind('\"')
    token = token_line[token_l + 1: token_r]

    t_min = float(find_nums(lines[1])[0])
    t_max = float(find_nums(lines[2])[0])
    return token, (t_max - t_min)


def calculate_durations(textgrid, phone2idx=None):
    tokens = []
    durs = []
    keyword = "words" if args.word_durations else "phones"

    with open(textgrid, 'r') as f:
        # Read file and get rid of header
        lines = [line.strip() for line in f.readlines()]
        idx = lines.index(f'name = "{keyword}"')
        total_frames = ceil(
            float(find_nums(lines[idx + 2])[0]) * args.sr / args.window_stride
        )
        total_tokens = int(find_nums(lines[idx + 3])[0])

        for i in range(idx + 4, idx + (total_tokens * 4), 4):
            token, dur = get_token_and_dur(lines[i: i+4])
            if phone2idx:
                tokens.append(phone2idx[token])
            else:
                tokens.append(token)
            durs.append(dur)

    durs = np.array(durs)
    durs *= (args.sr / args.window_stride)
    durs = np.rint(durs)

    # Take care of rounding error (may need extra space token)
    if phone2idx:
        tokens.append(phone2idx['sp'])
        tokens = np.array(tokens)
    else:
        tokens.append('')
    durs = np.append(durs, 0)

    if durs.sum() < total_frames:
        # Add silence frames
        durs[-1] = total_frames - durs.sum()
    elif durs.sum() > total_frames:
        # Remove frames from longest dur token
        longest_dur_token = np.argmax(durs)
        durs[longest_dur_token] -= durs.sum() - total_frames

    assert durs.sum() == total_frames
    assert len(durs) == len(tokens)

    return tokens, durs


def main():
    textgrid_list = glob.glob(os.path.join(args.ljspeech_dir, 'alignments/wavs/*.TextGrid'))

    # Create target_dir if necessary
    target_dir = ''
    if args.word_durations:
        target_dir = os.path.join(args.ljspeech_dir, 'word_durations/')
        print(f"Calculating word durations, files will be in: {target_dir}")
    else:
        target_dir = os.path.join(args.ljspeech_dir, 'phoneme_durations/')
        print(f"Calculating phoneme durations, files will be in: {target_dir}")

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    # Read phoneme to idx mappings
    phone2idx = None
    if args.mappings:
        with open(args.mappings, 'r') as f:
            phone2idx = json.load(f)['phone2idx']

    # Iterate through all TextGrid files
    for textgrid in textgrid_list:
        tokens, durs = calculate_durations(textgrid, phone2idx=phone2idx)
            
        basename = os.path.splitext(os.path.basename(textgrid))[0][5:]  # Chop off 'wavs_' prefix
        target_path = os.path.join(target_dir, f'{basename}.npz')
        np.savez(target_path, tokens=tokens, durs=durs)

if __name__ == '__main__':
    main()

