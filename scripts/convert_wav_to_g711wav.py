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
#
# USAGE:
# python convert_wav_to_g711wav.py \
#   --data_dir=<dir with .wav files> \
#   --dest_dir=<destination dir root>
#
# Converts all wav audio files to PCM u-law wav files (8kHz, 8-bit).
# Requires sox to be installed.
import argparse
import concurrent.futures
import glob
import logging
import os
import subprocess

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Convert wav audio to pcm mulaw wav')
parser.add_argument(
    "--data_dir", default=None, type=str, required=True, help="The path to the input directory with .wav files.",
)
parser.add_argument(
    "--dest_dir", default=None, type=str, required=True, help="Path to the destination directory.",
)
args = parser.parse_args()


def __convert_audio(in_path, out_path):
    """
    Helper function that's called per thread, converts wav to G.711 wav.
    Args:
        in_path: source wav file to convert
        out_path: destination for G.711 wav file
    """
    cmd = ["sox", in_path, "-r", "8000", "-c", "1", "-e", "u-law", out_path]
    subprocess.run(cmd)


def __process_set(data_dir, dst_root):
    """
    Finds and converts all wav audio files in the given directory to pcm_mulaw.
    Args:
        data_dir: source directory with wav files to convert
        dst_root: where G.711 (pcm_mulaw) wav files will be stored
    """
    wav_list = glob.glob(data_dir)

    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    # Set up and execute concurrent audio conversion
    tp = concurrent.futures.ProcessPoolExecutor(max_workers=64)
    futures = []

    for wav_path in tqdm(wav_list, desc="Submitting wav futures", unit="file"):
        audio_id = os.path.basename(wav_path)
        out_path = os.path.join(dst_root, audio_id)
        futures.append(tp.submit(__convert_audio, wav_path, out_path))

    pbar = tqdm(total=len(wav_list), desc="Converting wav files", unit="file")
    count = 0
    for f in concurrent.futures.as_completed(futures):
        count += 1
        pbar.update()
    tp.shutdown()
    pbar.close()


def main():
    data_dir = args.data_dir
    dest_dir = args.dest_dir

    logging.info("\n\nConverting audio in {}", data_dir)
    __process_set(
        os.path.join(data_dir, "*.wav",), os.path.join(dest_dir),
    )


if __name__ == '__main__':
    main()
