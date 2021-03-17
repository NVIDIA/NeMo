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
import sys

from nemo.collections.tts.data.datalayers import preprocess_linear_specs_dataset

sys.path.insert(0, '../')


def main():
    parser = argparse.ArgumentParser(
        description='Create dataset fitted for training and validating deep griffin iteration from wavefiles'
    )
    parser.add_argument(
        "-v",
        "--valid_filelist",
        help="Filelist for validation set, with all validation audio files listed",
        required=True,
        default=None,
        type=str,
    )
    parser.add_argument(
        "-t",
        "--train_filelist",
        help="Filelist for train set, with all train audio files listed",
        required=True,
        default=None,
        type=str,
    )
    parser.add_argument(
        "-n",
        "--n_fft",
        help="Value for the n_fft parameter, and the filter length for the STFT",
        default=512,
        type=int,
    )
    parser.add_argument("--hop_length", help="STFT parameter", default=256, type=int)
    parser.add_argument(
        "-d", "--destination", help="Destination to save the preprocessed data set to", default="/tmp", type=str
    )
    parser.add_argument(
        "-s",
        "--num_snr",
        help="Number of distinctive noisy samples to generate for each clear sample at the file list",
        default=1,
        type=int,
    )

    args = parser.parse_args()

    preprocess_linear_specs_dataset(
        args.valid_filelist, args.train_filelist, args.n_fft, args.hop_length, args.num_snr, args.destination
    )


if __name__ == "__main__":
    main()
