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
# USAGE: python get_data.py --data-root=<where to put data> --num-workers=<number of parallel workers> [--use-mic2]
# We have two audio files associated with each text due to the use of two microphones.
# Your call on whether to use the extra data. Data split takes this into account to prevent data leakage.

import argparse
import fnmatch
import functools
import json
import multiprocessing
import os
import subprocess
import zipfile
import urllib.request
from pathlib import Path
from nemo_text_processing.text_normalization.normalize import Normalizer
import random
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Download VCTK-Corpus and create manifests')
parser.add_argument("--data-root", required=True, type=Path)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument('--use_mic2', required=False, action='store_true')
args = parser.parse_args()

URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"


def __maybe_download_file(source_url, destination_path):
    if not destination_path.exists():
        tmp_file_path = destination_path.with_suffix('.tmp')
        urllib.request.urlretrieve(source_url, filename=str(tmp_file_path))
        tmp_file_path.rename(destination_path)


def __extract_file(filepath, data_dir):
    try:
        with zipfile.ZipFile(filepath) as zipped:
            zipped.extractall(data_dir)
    except Exception:
        print(f"Error while extracting {filepath}. Already extracted?")


def __process_transcript(file_pairs, text_normalizer):
    entries = []
    with open(file_pairs[1], encoding="utf-8") as fin:
        text = fin.readlines()[0].strip()

        wav_file = file_pairs[2]
        speaker_id = file_pairs[1].split('/')[-2]
        if not os.path.exists(wav_file):
            print(f"SKIPPED: {wav_file} not found!")
        else:
            duration = subprocess.check_output(f"soxi -D {wav_file}", shell=True)
            entry = {
                'audio_filepath': os.path.abspath(wav_file),
                'duration': float(duration),
                'text': text,
                'normalized_text': text_normalizer.normalize(text, punct_pre_process=True, punct_post_process=True),
                'speaker': str(speaker_id),
                'file_id': str(file_pairs[0])
            }
            entries.append(entry)

    return entries

def __process_data(data_folder, use_mic2, num_workers):
    text_normalizer = Normalizer(
        lang="en",
        input_case="cased",
        overwrite_cache=True,
        cache_dir=os.path.join(data_folder, "cache_dir"),
    )

    file_pairs = []
    entries = []
    all_file_ids = []

    for speaker in os.listdir(os.path.join(data_folder, "txt")):
        if speaker == ".DS_Store":
            continue
        for text_file in os.listdir(os.path.join(data_folder, "txt", speaker)):
            file_id = text_file.split(".")[0]
            all_file_ids.append(file_id)
            text_file = os.path.join(data_folder, "txt", speaker, text_file)
            wav_file = os.path.join(data_folder, "wav48_silence_trimmed", speaker, f"{file_id}_mic1.flac")
            file_pairs.append((file_id, text_file, wav_file))
            if use_mic2:
                wav_file = os.path.join(data_folder, "wav48_silence_trimmed", speaker, f"{file_id}_mic2.flac")
                file_pairs.append((file_id, text_file, wav_file))

    with multiprocessing.Pool(num_workers) as p:
        processing_func = functools.partial(__process_transcript, text_normalizer=text_normalizer)
        results = p.imap(processing_func, file_pairs)
        for result in tqdm(results, total=len(file_pairs)):
            entries.extend(result)

    # TODO: We're just doing a random split across all data (and hence all speakers).
    # You might want to take a look at this in the future.
    # For now we just take a random 10% each as validation data and test data
    # Note that we're basing this off the file ID since each text can have two audio files
    # We split based on ID to avoid data leakage
    random.seed(42)
    random.shuffle(all_file_ids)
    validation_ids = all_file_ids[:int(len(all_file_ids) * 0.1)]
    test_ids = all_file_ids[len(validation_ids):len(validation_ids)+int(len(all_file_ids) * 0.1)]

    test_outfile = os.path.join(data_folder, "test_manifest.json")
    val_outfile = os.path.join(data_folder, "validation_manifest.json")
    train_outfile = os.path.join(data_folder, "train_manifest.json")
    with open(test_outfile, 'w') as test_out, open(val_outfile, 'w') as val_out, open(train_outfile, 'w') as train_out:
        for m in entries:
            if m['file_id'] in test_ids:
                test_out.write(json.dumps(m) + '\n')
            elif m['file_id'] in validation_ids:
                val_out.write(json.dumps(m) + '\n')
            else:
                train_out.write(json.dumps(m) + '\n')


def main():
    data_root = args.data_root
    num_workers = args.num_workers
    use_mic2 = args.use_mic2
    filepath = data_root / f"vctk-corpus-0.92.zip"
    
    print(f"Downloading VCTK-Corpus data...")
    __maybe_download_file(URL, filepath)
    print("Extracting...")
    data_root = os.path.join(data_root, "VCTKCorpus-0.92")
    if not os.path.exists(data_root):
        os.mkdir(data_root)
    __extract_file(str(filepath), str(data_root))

    print("Processing and building manifest.")
    __process_data(
        data_root,
        num_workers=num_workers,
        use_mic2=use_mic2
    )



if __name__ == "__main__":
    main()
