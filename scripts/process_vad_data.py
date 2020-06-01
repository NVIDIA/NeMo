# Copyright 2020 NVIDIA. All Rights Reserved.
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

import argparse
import glob
import json
import logging
import os
import random
import tarfile
import urllib.request

import librosa
from sklearn.model_selection import train_test_split

sr = 16000
duration_stride = 1.0

# google speech command v2
URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


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


def extract_file(filepath: str, data_dir: str):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        logging.info('Not extracting. Maybe already there?')


def __extract_all_files(filepath: str, data_root: str, data_dir: str):
    if not os.path.exists(data_dir):
        extract_file(filepath, data_dir)
    else:
        logging.info(f'Skipping extracting. Data already there {data_dir}')


def split_train_val_test(data_dir, file_type, test_size=0.1, val_size=0.1):
    X = []
    for o in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, o)) and o.split("/")[-1] != "_background_noise_":
            X.extend(glob.glob(os.path.join(data_dir, o) + '/*.wav'))

    X_train, X_test = train_test_split(X, test_size=test_size, random_state=1)
    val_size_tmp = val_size / (1 - test_size)
    X_train, X_val = train_test_split(X_train, test_size=val_size_tmp, random_state=1)

    with open(os.path.join(data_dir, file_type + "_training_list.txt"), "w") as outfile:
        outfile.write("\n".join(X_train))
    with open(os.path.join(data_dir, file_type + "_testing_list.txt"), "w") as outfile:
        outfile.write("\n".join(X_test))
    with open(os.path.join(data_dir, file_type + "_validation_list.txt"), "w") as outfile:
        outfile.write("\n".join(X_val))

    logging.info(f'Overall: {len(X)}, Train: {len(X_train)}, Validatoin: {len(X_val)}, Test: {len(X_test)}')
    logging.info(f"Finished split train, val and test for {file_type}. Write to files !")


def process_google_speech_train(data_dir):
    X = []
    for o in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, o)) and o.split("/")[-1] != "_background_noise_":
            X.extend(glob.glob(os.path.join(data_dir, o) + '/*.wav'))

    short_files = [i.split(data_dir)[1] for i in files]

    with open(os.path.join(data_dir, 'testing_list.txt'), 'r') as allfile:
        testing_list = allfile.read().splitlines()

    with open(os.path.join(data_dir, 'validation_list.txt'), 'r') as allfile:
        validation_list = allfile.read().splitlines()

    exist_set = set(testing_list).copy()
    exist_set.update(set(validation_list))

    training_list = [i for i in short_files if i not in exist_set]

    with open(os.path.join(data_dir, "training_list.txt"), "w") as outfile:
        outfile.write("\n".join(training_list))

    logging.info(
        f'Overall: {len(files)}, Train: {len(training_list)}, Validatoin: {len(validation_list)}, Test: {len(testing_list)}'
    )


def write_manifest(
    data_dir,
    out_dir,
    files,
    prefix,
    manifest_name,
    duration_stride=1.0,
    duration_max=None,
    duration_limit=10.0,
    filter_long=False,
):
    seg_num = 0
    skip_num = 0
    if duration_max is None:
        duration_max = 1e9

    if not os.path.exists(out_dir):
        logging.info(f'Outdir {out_dir} does not exist. Creat directory.')
        os.mkdir(out_dir)

    output_path = os.path.join(out_dir, manifest_name + '.json')
    with open(output_path, 'w') as fout:
        for file in files:

            label = prefix

            try:
                x, _sr = librosa.load(file, sr=sr)
                duration = librosa.get_duration(x, sr=sr)

            except Exception:
                continue

            if filter_long and duration > duration_limit:
                skip_num += 1
                continue

            offsets = []
            durations = []

            if duration > duration_max:
                current_offset = 0.0

                while current_offset < duration:
                    difference = duration - current_offset
                    segment_duration = min(duration_max, difference)

                    offsets.append(current_offset)
                    durations.append(segment_duration)

                    current_offset += duration_stride

            else:
                offsets.append(0.0)
                durations.append(duration)

            for duration, offset in zip(durations, offsets):
                metadata = {
                    'audio_filepath': file,
                    'duration': duration,
                    'label': label,
                    'text': '_',  # for compatibility with ASRAudioText
                    'offset': offset,
                }
                json.dump(metadata, fout)
                fout.write('\n')
                fout.flush()
                seg_num += 1
    return skip_num, seg_num, output_path


def load_list_write_manifest(
    data_dir, out_dir, filename, prefix, duration_stride=1.0, duration_max=1.0, duration_limit=100.0, filter_long=True
):

    filename = prefix + '_' + filename
    file_path = os.path.join(data_dir, filename)

    with open(file_path, 'r') as allfile:
        files = allfile.read().splitlines()

    manifest_name = filename.split('_list.txt')[0] + '_manifest'
    skip_num, seg_num, output_path = write_manifest(
        data_dir,
        out_dir,
        files,
        prefix,
        manifest_name,
        duration_stride,
        duration_max,
        duration_limit,
        filter_long=True,
    )
    return skip_num, seg_num, output_path


def get_max_json(data_dir, data_json, max_limit, prefix):
    data = []
    seg = 0
    for line in open(data_json, 'r'):
        data.append(json.loads(line))
    filename = data_json.split('/')[-1]

    fout_path = os.path.join(data_dir, prefix + "_" + filename)
    selected_sample = random.sample(data, max_limit)

    with open(fout_path, 'a') as fout:
        for i in selected_sample:
            seg += 1
            json.dump(i, fout)
            fout.write('\n')
            fout.flush()

    logging.info(f'Get {seg}/{max_limit} to  {fout_path} from {data_json}')
    return fout_path


def main():
    parser = argparse.ArgumentParser(description='Speech and backgound data download and preprocess')
    parser.add_argument("--out_dir", required=True, default='./manifest/', type=str)
    parser.add_argument("--speech_data_root", required=True, default=None, type=str)
    parser.add_argument("--background_data_root", required=True, default=None, type=str)
    parser.add_argument('--test_size', required=False, default=0.1, type=float)
    parser.add_argument('--val_size', required=False, default=0.1, type=float)
    parser.add_argument('--log', required=False)
    parser.add_argument('--rebalance', required=False)
    parser.set_defaults(log=True, rebalance=False)

    args = parser.parse_args()

    if args.log:
        logging.basicConfig(level=logging.DEBUG)

    # Download speech data
    speech_data_root = args.speech_data_root
    data_set = "google_speech_recognition_v2"
    speech_data_folder = os.path.join(speech_data_root, data_set)

    background_data_folder = args.background_data_root
    logging.info(f"Working on: {data_set}")

    # Download and extract speech data
    if not os.path.exists(speech_data_folder):
        file_path = os.path.join(speech_data_root, data_set + ".tar.bz2")
        logging.info(f"Getting {data_set}")
        __maybe_download_file(file_path, URL)
        logging.info(f"Extracting {data_set}")
        __extract_all_files(file_path, speech_data_root, speech_data_folder)

    logging.info(f"Split speech data!")
    # dataset provide testing.txt and validation.txt feel free to split data using that with process_google_speech_train
    split_train_val_test(speech_data_folder, "speech", args.test_size, args.val_size)

    logging.info(f"Split background data!")
    split_train_val_test(background_data_folder, "background", args.test_size, args.val_size)

    out_dir = args.out_dir
    # Process Speech manifest
    logging.info(f"=== Write speech data to manifest!")
    skip_num_val, seg_num_val, speech_val = load_list_write_manifest(
        speech_data_folder, out_dir, 'validation_list.txt', 'speech', 1, 1
    )
    skip_num_test, seg_num_test, speech_test = load_list_write_manifest(
        speech_data_folder, out_dir, 'testing_list.txt', 'speech', 1, 1
    )
    skip_num_train, seg_num_train, speech_train = load_list_write_manifest(
        speech_data_folder, out_dir, 'training_list.txt', 'speech', 1, 1
    )

    logging.info(f'Val: Skip {skip_num_val} samples. Get {seg_num_val} segments! => {speech_val} ')
    logging.info(f'Test: Skip {skip_num_test} samples. Get {seg_num_test} segments! => {speech_test}')
    logging.info(f'Train: Skip {skip_num_train} samples. Get {seg_num_train} segments!=> {speech_train}')
    min_seg_num_val = seg_num_val
    min_seg_num_test = seg_num_test
    min_seg_num_train = seg_num_train

    # Process background manifest
    logging.info(f"=== Write background data to manifest!")
    skip_num_val, seg_num_val, background_val = load_list_write_manifest(
        background_data_folder, out_dir, 'validation_list.txt', 'background', 1, 1
    )
    skip_num_test, seg_num_test, background_test = load_list_write_manifest(
        background_data_folder, out_dir, 'testing_list.txt', 'background', 1, 1
    )
    skip_num_train, seg_num_train, background_train = load_list_write_manifest(
        background_data_folder, out_dir, 'training_list.txt', 'background', 1, 1
    )

    logging.info(f'Val: Skip {skip_num_val} samples. Get {seg_num_val} segments! => {background_val}')
    logging.info(f'Test: Skip {skip_num_test} samples. Get {seg_num_test} segments! => {background_test}')
    logging.info(f'Train: Skip {skip_num_train} samples. Get {seg_num_train} segments! =>{background_train}')
    min_seg_num_val = min(min_seg_num_val, seg_num_val)
    min_seg_num_test = min(min_seg_num_test, seg_num_test)
    min_seg_num_train = min(min_seg_num_train, seg_num_train)

    logging.info('Done!')

    if args.rebalance:
        # Get balanced amount of data in both classes.
        logging.info("Rebalancing number of samples in classes.")
        logging.info(f'Val: {min_seg_num_val} Test: {min_seg_num_test} Train: {min_seg_num_train}!')

        get_max_json(out_dir, background_val, min_seg_num_val, 'balanced')
        get_max_json(out_dir, background_test, min_seg_num_test, 'balanced')
        get_max_json(out_dir, background_train, min_seg_num_train, 'balanced')

        get_max_json(out_dir, speech_val, min_seg_num_val, 'balanced')
        get_max_json(out_dir, speech_test, min_seg_num_test, 'balanced')
        get_max_json(out_dir, speech_train, min_seg_num_train, 'balanced')
    else:
        logging.info("Don't rebalance number of samples in classes.")


if __name__ == '__main__':
    main()
