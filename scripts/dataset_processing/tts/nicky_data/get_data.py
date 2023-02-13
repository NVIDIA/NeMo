# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#
# USAGE: python get_data.py --data-file=<path to nicky data> --data-root=<where to put data> --num-workers=<number of parallel workers>
# Example: python get_data.py --data-file data/nicky_data.zip --data-root data/ --num-workers=4
# 
# NickyData Setup Guide:
# 1. Download the `txt` and `wav48` folders from Google Drive and place the zip file in your instance.
# 2. Run this script.
# 3. Set the following params in `inpainting.yaml`: 
# ```
# train_dataset:  data/NickyData/train_manifest.json
# validation_datasets: data/NickyData/validation_manifest.json
# sup_data_path: data/NickyData/data_cache
# ```

import argparse
import functools
import json
import multiprocessing
import os
import subprocess
import zipfile
from pathlib import Path
from nemo_text_processing.text_normalization.normalize import Normalizer
import sox
from tqdm import tqdm
import random

parser = argparse.ArgumentParser(description='Extract Nicky Data and create manifests')
parser.add_argument("--data-file", required=True, type=Path)
parser.add_argument("--data-root", required=True, type=Path)
parser.add_argument("--num-workers", default=4, type=int)
args = parser.parse_args()

def __extract_file(filepath, data_dir):
    try:
        with zipfile.ZipFile(filepath) as zipped:
            zipped.extractall(data_dir)
    except Exception:
        print(f"Error while extracting {filepath}. Already extracted?")


def __process_transcript(file_pairs, text_normalizer):
    entries = []
    with open(file_pairs[0], encoding="utf-8") as fin:
        text = fin.readlines()[0].strip()

        wav_file = file_pairs[1]
        speaker_id = file_pairs[0].split('/')[-2]
        if not os.path.exists(wav_file):
            print(f"SKIPPED: {wav_file} not found!")
        # TODO: Need to figure out why this file bugs out
        # Also setup a better data pipeline to ignore all cases like this
        elif wav_file.split("/")[-1] in ["p400_25.wav", "p400_636.wav"]:
            print(f"SKIPPED DUE TO BUG: {wav_file}")
        else:
            duration = subprocess.check_output(f"soxi -D {wav_file}", shell=True)
            entry = {
                'audio_filepath': os.path.abspath(wav_file),
                'duration': float(duration),
                'text': text,
                'normalized_text': text_normalizer.normalize(text, punct_pre_process=True, punct_post_process=True),
                'speaker': str(speaker_id),
            }
            entries.append(entry)

    return entries


def __process_data(data_folder, num_workers):
    text_normalizer = Normalizer(
        lang="en",
        input_case="cased",
        overwrite_cache=True,
        cache_dir=os.path.join(data_folder, "cache_dir"),
    )

    file_pairs = []
    entries = []

    for speaker in os.listdir(os.path.join(data_folder, "txt")):
        if speaker == ".DS_Store":
            continue
        for text_file in os.listdir(os.path.join(data_folder, "txt", speaker)):
            file_id = int(text_file.split("_")[1].split(".")[0])
            wav_file = os.path.join(data_folder, "wav48", speaker, f"{text_file.split('_')[0]}_{file_id}.wav")
            text_file = os.path.join(data_folder, "txt", speaker, text_file)
            file_pairs.append((text_file, wav_file))

    with multiprocessing.Pool(num_workers) as p:
        processing_func = functools.partial(__process_transcript, text_normalizer=text_normalizer)
        results = p.imap(processing_func, file_pairs)
        for result in tqdm(results, total=len(file_pairs)):
            entries.extend(result)

    # TODO: If/when Nicky data has other speakers, this needs to change
    # For now we just take a random 10% each as validation data and test data
    random.seed(42)
    random.shuffle(entries)
    validation_size = int(len(entries) * 0.1)
    test_size = int(len(entries) * 0.1)
    
    with open(os.path.join(data_folder, "test_manifest.json"), 'w') as fout:
        for m in entries[:test_size]:
            fout.write(json.dumps(m) + '\n')

    with open(os.path.join(data_folder, "validation_manifest.json"), 'w') as fout:
        for m in entries[test_size:test_size+validation_size]:
            fout.write(json.dumps(m) + '\n')

    with open(os.path.join(data_folder, "train_manifest.json"), 'w') as fout:
        for m in entries[test_size+validation_size:]:
            fout.write(json.dumps(m) + '\n')

def main():
    data_root = args.data_root
    num_workers = args.num_workers
    filepath = args.data_file
    
    print("Extracting...")
    data_root = os.path.join(data_root, "NickyData")
    if not os.path.exists(data_root):
        os.mkdir(data_root)
    __extract_file(str(filepath), str(data_root))

    print("Processing and building manifest.")
    __process_data(
        data_root,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()
