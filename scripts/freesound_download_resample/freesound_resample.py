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
import argparse
import glob
import os
import time

import librosa
import sox
from joblib import Parallel, delayed


def resample_file(resampled_dir, filepath, ext, sample_rate):
    """
    Resample an audio file to 16kHZ and transform to monochannel
    Remove incompatible files.

    Args:
        resampled_dir: Directory of transformed files.
        filepath: Filepath of Audio
        ext: File type e.g. "wav", "flac"

    Returns:

    """
    head, filename = os.path.split(filepath)
    _, clsname = os.path.split(head)

    filename, _ = os.path.splitext(filename)

    new_dir = os.path.join(resampled_dir, clsname)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    new_path = os.path.join(new_dir, filename + f'.{ext}')

    # check if the resampled data exists.
    if os.path.exists(new_path):
        print(f"Resampled file {filepath} exists. Skip it.")
        return None

    transform = sox.Transformer()
    transform.set_output_format(file_type='wav')
    transform.convert(samplerate=sample_rate, n_channels=1)

    try:
        transform.build(filepath, new_path)
        print(f"Finished converting file {filepath}.")

        return None

    except sox.core.SoxError as e:

        try:
            # Check if the file is readable
            librosa.load(path=filepath)

            # if it is, force input format and try again
            transform.set_input_format(file_type=ext)
            transform.build(filepath, new_path)
            return None

        except Exception:
            return filepath


def main():
    start = time.time()
    parser = argparse.ArgumentParser(description='Freesound data resample')
    parser.add_argument("--data_dir", required=True, default=None, type=str)
    parser.add_argument('--resampled_dir', required=True, default=None, type=str)
    parser.add_argument('--sample_rate', default=16000, type=int)
    args = parser.parse_args()

    data_dir = args.data_dir
    resampled_dir = args.resampled_dir
    sample_rate = args.sample_rate

    wav_files = sorted(glob.glob(os.path.join(data_dir, '*/*.wav')))
    flac_files = sorted(glob.glob(os.path.join(data_dir, '*/*.flac')))

    with Parallel(n_jobs=-1, verbose=10) as parallel:
        wav_files_failed = parallel(
            delayed(resample_file)(resampled_dir, filepath, ext='wav', sample_rate=sample_rate)
            for filepath in wav_files
        )

        flac_files_failed = parallel(
            delayed(resample_file)(resampled_dir, filepath, ext='flac', sample_rate=sample_rate)
            for filepath in flac_files
        )

    with open('dataset_conversion_logs.txt', 'w') as f:
        for file in wav_files_failed:
            if file is not None:
                f.write(f"{file}\n")

        for file in flac_files_failed:
            if file is not None:
                f.write(f"{file}\n")

    end = time.time()
    print(f'Resample data in {data_dir} and save to {resampled_dir} takes {end-start} seconds.')


if __name__ == '__main__':

    main()
