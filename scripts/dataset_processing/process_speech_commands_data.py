# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
Usage:

python process_speech_commands_data.py \
    --data_root=<absolute path to where the data should be stored> \
    --data_version=<either 1 or 2, indicating version of the dataset> \
    --class_split=<either "all" or "sub", indicates whether all 30/35 classes should be used, or the 10+2 split should be used> \
    --rebalance \
    --log
"""

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
        extract_file(filepath, data_dir)
    else:
        logging.info(f'Skipping extracting. Data already there {data_dir}')


def extract_file(filepath: str, data_dir: str):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        logging.info('Not extracting. Maybe already there?')


def __process_data(
    data_folder: str, dst_folder: str, rebalance: bool = False, class_split: str = "all", skip_duration: bool = False
):
    """
    To generate manifest

    Args:
        data_folder: source with wav files and validation / test lists
        dst_folder: where manifest files will be stored
        rebalance: rebalance the classes to have same number of samples.
        class_split: whether to use all classes as distinct labels, or to use
            10 classes subset and rest of the classes as noise or background.
        skip_duration: Bool whether to skip duration computation. Use this only for
            colab notebooks where knowing duration is not necessary for demonstration.

    Returns:

    """

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Used for 10 classes + silence + unknown class setup
    class_subset = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

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

    logging.info("Validation and test set lists extracted")

    label_count = {}
    label_filepaths = {}
    unknown_val_filepaths = []
    unknown_test_filepaths = []

    train, val, test = [], [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)

            if label == '_background_noise_' or label == 'silence':
                continue

            if class_split == "sub":
                if label not in class_subset:
                    label = "unknown"  # replace label

                    if uid in valset:
                        unknown_val_filepaths.append((label, entry))
                    elif uid in testset:
                        unknown_test_filepaths.append((label, entry))

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

            if label == 'unknown':
                continue

            if uid in valset:
                val.append(sample)
            elif uid in testset:
                test.append(sample)
            else:
                train.append(sample)

    logging.info("Prepared filepaths for dataset")

    # Add silence and unknown class label samples
    if class_split == "sub":
        logging.info("Perforiming 10+2 class subsplit")

        silence_path = os.path.join(data_folder, "silence")
        if not os.path.exists(silence_path):
            os.mkdir(silence_path)

        silence_stride = 1000  # 0.0625 second stride
        sampling_rate = 16000
        folder = os.path.join(data_folder, "_background_noise_")

        silence_files = []
        rng = np.random.RandomState(0)

        for file in os.listdir(folder):
            if ".wav" in file:
                load_path = os.path.join(folder, file)
                y, sr = librosa.load(load_path, sr=sampling_rate)

                for i in range(0, len(y) - sampling_rate, silence_stride):
                    file_path = "silence/{}_{}.wav".format(file[:-4], i)
                    y_slice = y[i : i + sampling_rate]
                    magnitude = rng.uniform(0.0, 1.0)
                    y_slice *= magnitude
                    out_file_path = os.path.join(data_folder, file_path)
                    librosa.output.write_wav(out_file_path, y_slice, sr)

                    silence_files.append(('silence', out_file_path))

        rng = np.random.RandomState(0)
        rng.shuffle(silence_files)
        logging.info(f"Constructed silence set of {len(silence_files)}")

        # Create the splits
        rng = np.random.RandomState(0)
        silence_split = 0.1
        unknown_split = 0.1

        # train split
        num_total_samples = sum([label_count[cls] for cls in class_subset])
        num_silence_samples = int(np.ceil(silence_split * num_total_samples))

        # initialize sample
        label_count['silence'] = 0

        for silence_id in range(num_silence_samples):
            label_count['silence'] += 1

            if 'silence' in label_filepaths:
                label_filepaths['silence'] += [silence_files[silence_id]]
            else:
                label_filepaths['silence'] = [silence_files[silence_id]]

        train.extend(label_filepaths['silence'])

        # Update train unknown set
        unknown_train_samples = label_filepaths['unknown']

        rng.shuffle(unknown_train_samples)
        unknown_size = int(np.ceil(unknown_split * num_total_samples))

        label_count['unknown'] = unknown_size
        label_filepaths['unknown'] = unknown_train_samples[:unknown_size]

        train.extend(label_filepaths['unknown'])

        logging.info("Train set prepared")

        # val set silence
        num_val_samples = len(val)
        num_silence_samples = int(np.ceil(silence_split * num_val_samples))

        val_idx = label_count['silence'] + 1
        for silence_id in range(num_silence_samples):
            val.append(silence_files[val_idx + silence_id])

        # Update val unknown set
        rng.shuffle(unknown_val_filepaths)
        unknown_size = int(np.ceil(unknown_split * num_val_samples))

        val.extend(unknown_val_filepaths[:unknown_size])

        logging.info("Validation set prepared")

        # test set silence
        num_test_samples = len(test)
        num_silence_samples = int(np.ceil(silence_split * num_test_samples))

        test_idx = val_idx + num_silence_samples + 1
        for silence_id in range(num_silence_samples):
            test.append(silence_files[test_idx + silence_id])

        # Update test unknown set
        rng.shuffle(unknown_test_filepaths)
        unknown_size = int(np.ceil(unknown_split * num_test_samples))

        test.extend(unknown_test_filepaths[:unknown_size])

        logging.info("Test set prepared")

    max_command = None
    max_count = -1
    for command, count in label_count.items():
        if command == 'unknown':
            continue

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

    manifests = [
        ('train_manifest.json', train),
        ('validation_manifest.json', val),
        ('test_manifest.json', test),
    ]

    for manifest_filename, dataset in manifests:
        with open(os.path.join(dst_folder, manifest_filename), 'w') as fout:
            num_files = len(dataset)
            pct_file = num_files // 100
            file_count = 0

            logging.info(f"Preparing manifest : {manifest_filename} with #{num_files} files")

            for label, audio_path in dataset:
                if not skip_duration:
                    duration = librosa.core.get_duration(filename=audio_path)
                else:
                    duration = 0.0

                # Write the metadata to the manifest
                metadata = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "command": label,
                }
                json.dump(metadata, fout)
                fout.write('\n')
                fout.flush()

                file_count += 1
                if file_count % pct_file == 0:
                    if not skip_duration:
                        logging.info(f"Finished serializing {file_count} / {num_files} into {manifest_filename}")

        logging.info(f"Finished construction of manifest : {manifest_filename}")

        if skip_duration:
            logging.info(
                f"\n<<NOTE>> Duration computation was skipped for demonstration purposes on Colaboratory.\n"
                f"In order to replicate paper results and properly perform data augmentation, \n"
                f"please recompute the manifest file without the `--skip_duration` flag !\n"
            )


def main():
    parser = argparse.ArgumentParser(description='Google Speech Command Data download')
    parser.add_argument("--data_root", required=True, default=None, type=str)
    parser.add_argument('--data_version', required=True, default=1, type=int, choices=[1, 2])
    parser.add_argument('--class_split', required=False, default='all', type=str, choices=['all', 'sub'])
    parser.add_argument('--rebalance', required=False, action='store_true')
    parser.add_argument('--skip_duration', required=False, action='store_true')
    parser.add_argument('--log', required=False, action='store_true')
    parser.set_defaults(log=False, rebalance=False, skip_duration=False)
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
    __process_data(
        data_folder,
        data_folder,
        rebalance=args.rebalance,
        class_split=args.class_split,
        skip_duration=args.skip_duration,
    )
    logging.info('Done!')


if __name__ == "__main__":
    main()
