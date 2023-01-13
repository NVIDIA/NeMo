# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
# USAGE: python get_librispeech_data.py --data_root=<where to put data>
#        --data_set=<datasets_to_download> --num_workers=<number of parallel workers>
# where <datasets_to_download> can be: dev_clean, dev_other, test_clean,
# test_other, train_clean_100, train_clean_360, train_other_500 or ALL
# You can also put more than one data_set comma-separated:
# --data_set=dev_clean,train_clean_100
import argparse
import fnmatch
import functools
import json
import logging
import multiprocessing
import os
import subprocess
import tarfile
import urllib.request

from sox import Transformer
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Generate manifest files from wavlist")
parser.add_argument("--wav_list", required=True, default=None, type=str)
parser.add_argument("--dest_json", required=True, default=None, type=str)
parser.add_argument("--num_workers", default=4, type=int)
args = parser.parse_args()

def __process_transcript(file_path: str):
    """
    Converts flac files to wav from a given transcript, capturing the metadata.
    Args:
        file_path: path to a source transcript  with flac sources
    Returns:
        a list of metadata entries for processed files.
    """
    entries = []
    wav_file = file_path
    transcript_text = "-"

    # Convert FLAC file to WAV
    if not os.path.exists(wav_file):
        raise ValueError(f"wav file does not exist: {wav_file}")
    # check duration
    duration = subprocess.check_output("soxi -D {0}".format(wav_file), shell=True)

    entry = {}
    entry["audio_filepath"] = os.path.abspath(wav_file)
    entry["duration"] = float(duration)
    entry["text"] = transcript_text
    
    entries.append(entry)
    return entries


def __process_data(wav_list_path: str, manifest_file: str, num_workers: int):
    """
    Converts flac to wav and build manifests's json
    Args:
        data_folder: source with flac files
        dst_folder: where wav files will be stored
        manifest_file: where to store manifest
        num_workers: number of parallel workers processing files
    Returns:
    """

    _files = open(wav_list_path, "r").readlines()
    files = [ x.strip() for x in _files ]
    entry = __process_transcript(files[0]) 
    entries = []

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.imap(__process_transcript, files)
        for result in tqdm(results, total=len(files)):
            entries.extend(result)

    with open(manifest_file, "w") as fout:
        for m in entries:
            fout.write(json.dumps(m) + "\n")


def main():
    wav_list = args.wav_list
    dest_json = args.dest_json
    num_workers = args.num_workers

    __process_data(
        wav_list_path=wav_list,
        manifest_file=dest_json,
        num_workers=num_workers,
    )
    logging.info("Done!")


if __name__ == "__main__":
    main()
