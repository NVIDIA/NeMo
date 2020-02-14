# Copyright (c) 2019 NVIDIA Corporation
#
# USAGE: python get_ljspeech_data.py --data_root=<where to put data>

import argparse
import glob
import json
import logging
import os
import re
import tarfile
import urllib.request

import librosa
import numpy as np

URL_v1 = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
URL_v2 = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


def __maybe_download_file(destination: str, source: str):
    """
    Downloads source to destination if it doesn't exist.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource

    Returns:

    """
    if not os.path.exists(destination):
        logging.info(f"{destination} does not exist. Downloading ...")
        urllib.request.urlretrieve(source, filename=destination + '.tmp')
        os.rename(destination + '.tmp', destination)
        logging.info(f"Downloaded {destination}.")
    else:
        logging.info(f"Destination {destination} exists. Skipping.")
    return destination


def __extract_all_files(filepath: str, data_root: str, data_dir: str):
    if not os.path.exists(data_dir):
        extract_file(filepath, data_root)
    else:
        logging.info(f'Skipping extracting. Data already there {data_dir}')


def extract_file(filepath: str, data_dir: str):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        logging.info('Not extracting. Maybe already there?')


def __process_data(data_folder: str, dst_folder: str, rebalance: bool = False):
    """
    To generate manifest

    Args:
        data_folder: source with wav files and validation / test lists
        dst_folder: where manifest files will be stored
        rebalance:

    Returns:

    """

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    pattern = re.compile(r"(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob.glob(os.path.join(data_folder, '*/*wav'))

    with open(os.path.join(data_folder, 'validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()

    valset = set()
    for entry in validation_files:
        r = re.match(pattern, entry)
        if r:
            valset.add(r.group(3))

    with open(os.path.join(data_folder, 'testing_list.txt'), 'r') as fin:
        testing_files = fin.readlines()

    testset = set()
    for entry in testing_files:
        r = re.match(pattern, entry)
        if r:
            testset.add(r.group(3))

    label_count = {}
    label_filepaths = {}

    train, val, test = [], [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                continue

            sample = (label, entry)

            if uid not in valset and uid not in testset:
                if label in label_count:
                    label_count[label] += 1
                else:
                    label_count[label] = 1

                if label in label_filepaths:
                    label_filepaths[label] += [sample]
                else:
                    label_filepaths[label] = [sample]

            if uid in valset:
                val.append(sample)
            elif uid in testset:
                test.append(sample)
            else:
                train.append(sample)

    max_command = None
    max_count = -1
    for command, count in label_count.items():
        if count > max_count:
            max_count = count
            max_command = command

    if rebalance:
        logging.info(f"Command with maximum number of samples = {max_command} with {max_count} samples")
        logging.info(f"Rebalancing dataset by duplicating classes with less than {max_count} samples...")

        for command, samples in label_filepaths.items():
            filepaths = [sample[1] for sample in samples]

            rng = np.random.RandomState(0)
            filepaths = np.asarray(filepaths)
            num_samples = len(filepaths)

            if num_samples < max_count:
                difference = max_count - num_samples
                duplication_ids = rng.choice(num_samples, difference, replace=True)

                filepaths = np.append(filepaths, filepaths[duplication_ids], axis=0)

                logging.info(f"Extended class label {command} from {num_samples} samples to {len(filepaths)} samples")

                label_filepaths[command] = [(command, filepath) for filepath in filepaths]

        del train
        train = []
        for label, samples in label_filepaths.items():
            train.extend(samples)

        print()

    manifests = [
        ('train_manifest.json', train),
        ('validation_manifest.json', val),
        ('test_manifest.json', test),
    ]

    for manifest_filename, dataset in manifests:
        with open(os.path.join(dst_folder, manifest_filename), 'w') as fout:
            for label, audio_path in dataset:
                duration = librosa.core.get_duration(filename=audio_path)

                # Write the metadata to the manifest
                metadata = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "command": label,
                }
                json.dump(metadata, fout)
                fout.write('\n')
                fout.flush()

        print(f"Finished construction of manifest : {manifest_filename}")


def main():
    parser = argparse.ArgumentParser(description='LJSpeech Data download')
    parser.add_argument("--data_root", required=True, default=None, type=str)
    parser.add_argument('--data_version', required=True, default=1, type=int, choices=[1, 2])
    parser.add_argument('--rebalance', required=False, action='store_true')
    parser.add_argument('--log', required=False, action='store_true')
    parser.set_defaults(log=False, rebalance=False)
    args = parser.parse_args()

    if args.log:
        logging.basicConfig(level=logging.DEBUG)

    data_root = args.data_root
    data_set = "google_speech_recognition_v{0}".format(args.data_version)
    data_folder = os.path.join(data_root, data_set)

    logging.info(f"Working on: {data_set}")

    if args.data_version == 1:
        URL = URL_v1
    else:
        URL = URL_v2

    # Download and extract
    if not os.path.exists(data_folder):
        file_path = os.path.join(data_root, data_set + ".tar.bz2")
        logging.info(f"Getting {data_set}")
        __maybe_download_file(file_path, URL)
        logging.info(f"Extracting {data_set}")
        __extract_all_files(file_path, data_root, data_folder)

    logging.info(f"Processing {data_set}")
    __process_data(data_root, data_folder, rebalance=args.rebalance)
    logging.info('Done!')


if __name__ == "__main__":
    main()
