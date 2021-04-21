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
import os
import pickle
from math import ceil

import numpy as np
import tgt
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Calculates phoneme durations for LJSpeech from TextGrids.")
parser.add_argument('--ljspeech_dir', required=True, default=None, type=str)
parser.add_argument(
    '--mappings',
    required=False,
    default=None,
    type=str,
    help='JSON file of mappings created with create_token2idx_dict.py',
)
parser.add_argument('--sr', required=False, default=22050, type=int)
parser.add_argument('--hop_length', required=False, default=256, type=int)
args = parser.parse_args()


def calculate_durations(textgrid, phone2idx):
    tokens = []
    durs = []

    frames_per_second = args.sr / args.hop_length
    tg = tgt.read_textgrid(textgrid, include_empty_intervals=True)
    data_tier = tg.get_tier_by_name("phones")

    # Get total frames
    total_frames = ceil((data_tier.end_time - data_tier.start_time) * frames_per_second)

    # Find start and end frames of each token
    se_in_frames = np.array([(frames_per_second * d.start_time, frames_per_second * d.end_time) for d in data_tier])
    se_in_frames = np.round(se_in_frames)
    durs = (se_in_frames[:, 1] - se_in_frames[:, 0]).astype(int)
    blank_set = ('sil', 'sp', 'spn', '', '<unk>')
    blank_token = " "

    # merge repeated blank tokens
    tokens, durations = [], []
    for i in range(len(data_tier)):
        x = data_tier[i].text
        if x == 'spn':
            return None, None, None
        x = blank_token if x in blank_set else x

        if len(tokens) and tokens[-1] == blank_token and x == blank_token:
            durations[-1] += durs[i]
        else:
            tokens.append(x)
            durations.append(durs[i])

    tokens_enc = [phone2idx[token] for token in tokens]
    tokens_enc, durations = torch.LongTensor(tokens_enc), torch.LongTensor(durations)

    # Add rounding error to final token
    durations[-1] += total_frames - durations.sum()

    return tokens, tokens_enc, durations


def main():
    textgrid_list = glob.glob(os.path.join(args.ljspeech_dir, 'alignments/wavs/*.TextGrid'))

    # Create target_dir if necessary
    target_dir = os.path.join(args.ljspeech_dir, 'phoneme_durations/')
    print(f"Calculating phoneme durations, files will be in: {target_dir}")

    if not os.path.exists(target_dir):
        print(f"Creating {target_dir}")
        os.mkdir(target_dir)

    # Read phoneme to idx mappings
    phone2idx = None
    if args.mappings:
        with open(args.mappings, 'r') as f:
            mappings = json.load(f)
            phone2idx = mappings['phone2idx']

    oov_samples = []

    # Iterate through all TextGrid files
    for textgrid in tqdm(textgrid_list):
        basename = os.path.splitext(os.path.basename(textgrid))[0][5:]  # Chop off 'wavs_' prefix

        phones_mfa, tokens_mfa, durs = calculate_durations(textgrid, phone2idx)

        if phones_mfa is None:
            oov_samples.append(basename)
            continue

        # Save to file
        target_path = os.path.join(target_dir, f'{basename}.pt')
        torch.save({'text_encoded': tokens_mfa, 'token_duration': durs}, target_path)

    print(f"Getting rid of {len(oov_samples)} samples with OOV words.")
    oov_target = os.path.join(args.ljspeech_dir, 'wavs_to_ignore.pkl')
    with open(oov_target, 'wb') as f:
        pickle.dump(oov_samples, f)
    print(f"List of OOV samples written to: {oov_target}")


if __name__ == '__main__':
    main()
