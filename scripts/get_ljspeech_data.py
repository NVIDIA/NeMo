# Copyright (c) 2019 NVIDIA Corporation
#
# USAGE: python get_ljspeech_data.py --data_root=<where to put data>

import argparse
import json
import os
import random
import tarfile
import urllib.request
from scipy.io.wavfile import read

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


def __maybe_download_file(destination: str, source: str):
    """
    Downloads source to destination if not exists.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource

    Returns:

    """
    source = URL
    if not os.path.exists(destination):
        print(f"{destination} does not exists. Downloading ...")
        urllib.request.urlretrieve(source, filename=destination + '.tmp')
        os.rename(destination + '.tmp', destination)
        print(f"Downloaded {destination}.")
    else:
        print(f"Destination {destination} exists. Skipping.")
    return destination


def __extract_all_files(filepath: str, data_root: str, data_dir: str):
    if not os.path.exists(data_dir):
        extract_file(filepath, data_root)
        audio_dir = os.path.join(data_dir, 'wav')
        for subfolder, _, filelist in os.walk(audio_dir):
            for ftar in filelist:
                extract_file(os.path.join(subfolder, ftar), subfolder)
    else:
        print(f'Skipping extracting. Data already there {data_dir}')


def extract_file(filepath: str, data_dir: str):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        print('Not extracting. Maybe already there?')


def __process_data(data_folder: str, dst_folder: str):
    """
    To generate manifest
    Args:
        data_folder: source with wav files
        dst_folder: where manifest files will be stored
    Returns:

    """

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    metadata_csv_path = os.path.join(data_folder, "metadata.csv")
    wav_folder = os.path.join(data_folder, "wavs")
    entries = []

    with open(metadata_csv_path) as f:
        line = f.readline()
        while line:
            file, _, transcript = line.split("|")
            wav_file = os.path.join(wav_folder, file + ".wav")
            sr, y = read(wav_file)
            assert sr == 22050
            duration = len(y) / sr

            entry = dict()
            entry['audio_filepath'] = os.path.abspath(wav_file)
            entry['duration'] = float(duration)
            entry['text'] = transcript
            entries.append(entry)
            line = f.readline()

    # Randomly split 64 samples from the entire dataset to create the
    # validation set
    random.shuffle(entries)
    training_set = entries[:-64]
    val_set = entries[-64:]
    with open(os.path.join(dst_folder, "ljspeech_train.json"), 'w') as fout:
        for m in training_set:
            fout.write(json.dumps(m) + '\n')
    with open(os.path.join(dst_folder, "ljspeech_eval.json"), 'w') as fout:
        for m in val_set:
            fout.write(json.dumps(m) + '\n')


def main():
    parser = argparse.ArgumentParser(description='LJSpeech Data download')
    parser.add_argument("--data_root", required=True, default=None, type=str)
    args = parser.parse_args()

    data_root = args.data_root
    data_set = "LJSpeech-1.1"
    data_folder = os.path.join(data_root, data_set)

    print(f"Working on: {data_set}")

    # Download and extract
    if not os.path.exists(data_folder):
        file_path = os.path.join(data_root, data_set + ".tar.bz2")
        print(f"Getting {data_set}")
        __maybe_download_file(file_path, data_set)
        print(f"Extracting {data_set}")
        __extract_all_files(file_path, data_root, data_folder)

    print(f"Processing {data_set}")
    __process_data(data_folder, data_folder)
    print('Done!')


if __name__ == "__main__":
    main()
