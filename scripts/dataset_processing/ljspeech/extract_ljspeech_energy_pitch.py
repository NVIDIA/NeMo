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
import librosa
import pysptk
import numpy as np
from pathlib import Path
from scipy.io import wavfile

tqdm = None

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    pass

parser = argparse.ArgumentParser(
    description="Extracts energies (L2-norm of STFT frame amplitudes) from LJSpeech data."
)
parser.add_argument("--ljspeech_dir", required=True, default=None, type=str)
args = parser.parse_args()


def main():
    wavfile_list = list(Path(args.ljspeech_dir + "wavs").glob('*.wav'))

    target_dir = Path(args.ljspeech_dir)
    # Create target dir <LJSpeech_base_dir>/energies and <LJSpeech_base_dir>/pitch if necessary
    if not Path(target_dir / "energies").exists():
        print(f"Creating target directory: {target_dir/'energies'}")
        Path(target_dir / "energies").mkdir()
    if not Path(target_dir / "pitch").exists():
        print(f"Creating target directory: {target_dir/'pitch'}")
        Path(target_dir / "pitch").mkdir()

    if tqdm is not None:
        wavfile_list = tqdm(wavfile_list)
    for count, file_ in enumerate(wavfile_list):
        basename = Path(file_).stem
        audio, sr = librosa.load(file_, sr=22050)
        fs, x = wavfile.read(str(file_))

        # Calculate f0
        f0 = pysptk.rapt(x.astype(np.float32) * 32768, fs=sr, hopsize=256, otype="f0")

        # Save to new file
        save_path = target_dir / "pitch" / f"{basename}.npy"
        np.save(save_path, f0)

        # Calculate energy
        stft_amplitude = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256, win_length=1024))
        energy = np.linalg.norm(stft_amplitude, axis=0)  # axis=0 since librosa.stft -> (freq bins, frames)

        # Save to new file
        save_path = target_dir / "energies" / f"{basename}.npy"
        np.save(save_path, energy)

        assert energy.shape == f0.shape
        if tqdm is None and count % 1000 == 0:
            print(f"Finished processing {count} wav files...")

    print(f"Finished energy extraction for a total of {len(wavfile_list)} wav files.")


if __name__ == '__main__':
    main()
