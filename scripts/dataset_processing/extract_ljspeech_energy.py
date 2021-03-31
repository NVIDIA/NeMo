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
Extracts energy data from LJSpeech wav files and writes them to a directory as LJxxx-xxxx.npy files.
Assuming that the wavs are located at `<LJSpeech_base_path>/wavs`, this script will write the
energy files to `<LJSpeech_base_path>/energies/`, creating the directory if necessary.

USAGE: python extract_ljspeech_energy.py --ljspeech_dir=<LJSpeech_base_path>
"""
import argparse
import glob
import os

import librosa
import numpy as np
import torch

parser = argparse.ArgumentParser(description="Extracts energies (L2-norm of STFT frame amplitudes) from LJSpeech data.")
parser.add_argument("--ljspeech_dir", required=True, default=None, type=str)
args = parser.parse_args()


def main():
    wavfile_list = glob.glob(os.path.join(args.ljspeech_dir, 'wavs/', '*.wav'))

    # Create target dir <LJSpeech_base_dir>/energies if necessary
    target_dir = os.path.join(args.ljspeech_dir, 'energies/')
    if not os.path.exists(target_dir):
        print(f"Creating target directory: {target_dir}")
        os.makedirs(target_dir)

    count = 0
    for wavfile in wavfile_list:
        basename, _ = os.path.splitext(os.path.basename(wavfile))
        audio, sr = librosa.load(wavfile)

        # Calculate energy
        stft_amplitude = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256, win_length=1024))
        energy = np.linalg.norm(stft_amplitude, axis=0)  # axis=0 since librosa.stft -> (freq bins, frames)

        # Save to new file
        save_path = os.path.join(target_dir, f"{basename}.npy")
        np.save(save_path, energy)

        count += 1
        if count % 1000 == 0:
            print(f"Finished processing {count} wav files...")

    print(f"Finished energy extraction for a total of {count} wav files.")


if __name__ == '__main__':
    main()
