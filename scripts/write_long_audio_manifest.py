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
import json
import logging
import multiprocessing
import os
from argparse import ArgumentParser
from itertools import repeat

import librosa
import numpy as np


"""
This script is designed for inference of frame level Voice Activity Detection (VAD) 

This script serves three goals:
    (1) Write audio files to manifest
    (2) Split audio file for avoiding CUDA memory issue
    (3) Take care of joint of seperate json line for an audio file

Usage:
python write_long_audio_manifest.py  --inp_dir=<FULL PATH OF FOLDER OF AUDIO FILES> --manifest_name=<NAME OF OUTPUT JSON FILE>  --split_duration=300 --time_length=0.63 --num_worker=10

"""


def write_manifest(file, args_func):
    """
    Given a list of files, write them to manifest with restrictions.
    Args:
        files : file to be processed
        label : label for audio snippet.
        split_duration : Max duration of each audio clip (each line in json)
        time_length : Used for taking care of joint.
                Length of window for generating the frame.
    Returns:
        res : list of generated metadata line of json for file
    """

    label = args_func['label']
    split_duration = args_func['split_duration']
    time_length = args_func['time_length']

    res = []
    try:
        sr = 16000
        x, _sr = librosa.load(file, sr=sr)
        duration = librosa.get_duration(x, sr=sr)

        left = duration
        current_offset = 0
        status = 'single'

        while left > 0:
            if left <= split_duration:
                if status == 'single':
                    write_duration = left
                    current_offset = 0
                else:
                    status = 'end'
                    write_duration = left + time_length
                    current_offset -= time_length
                offset_inc = left
                left = 0
            else:
                if status == 'start' or status == 'next':
                    status = 'next'
                else:
                    status = 'start'

                if status == 'start':
                    write_duration = split_duration
                    offset_inc = split_duration
                else:
                    write_duration = split_duration + time_length
                    current_offset -= time_length
                    offset_inc = split_duration + time_length

                left -= split_duration

            metadata = {
                'audio_filepath': file,
                'duration': write_duration,
                'label': label,
                'text': '_',
                'offset': current_offset,
            }
            res.append(metadata)

            current_offset += offset_inc

    except Exception as e:
        err_file = "error.log"
        with open(err_file, 'w') as fout:
            fout.write(file + ":" + str(e))

    return res


def main():
    parser = ArgumentParser()
    parser.add_argument("--inp_dir", type=str, required=True, help="(full path) folder of files to be processed")
    parser.add_argument(
        "--inp_list", type=str, help="(full path) a file contains NAME of files inside inp_dir to be processed"
    )
    parser.add_argument(
        "--out_dir", type=str, default=".", help="[Optional](full path) location to store generated json file"
    )
    parser.add_argument("--manifest_name", type=str, required=True, help="name of generated json file")
    parser.add_argument("--split_duration", type=int, required=True, help="max duration of each audio clip/line")
    parser.add_argument(
        "--time_length", type=float, default=0.63, help="[Optional] time length of segment, default is 0.63s"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="[Optional] number of workers for multiprocessing")
    args = parser.parse_args()

    if not args.inp_list:
        input_audios = []
        for root, dirs, files in os.walk(args.inp_dir):
            for basename in files:
                if basename.endswith('.wav'):
                    filename = os.path.join(root, basename)
                    input_audios.append(filename)
    else:
        name_list = np.loadtxt(args.inp_list, dtype='str')
        input_audios = [os.path.join(args.inp_dir, name + ".wav") for name in name_list]
    print(f"Number of wav files to be processed: {len(input_audios)}")

    output_path = os.path.join(args.out_dir, args.manifest_name + '.json')
    print(f"Save generate manifest to {output_path}!")
    if not os.path.exists(args.out_dir):
        logging.info(f'Outdir {args.out_dir} does not exist. Creat directory.')
        os.mkdir(args.out_dir)
    if os.path.exists(output_path):
        print(f"Manifest {output_path} exists! Overwriting")
        os.remove(output_path)

    p = multiprocessing.Pool(processes=args.num_workers)

    args_func = {
        'label': 'infer',
        'split_duration': args.split_duration,
        'time_length': args.time_length,
    }

    results = p.starmap(write_manifest, zip(input_audios, repeat(args_func)))

    with open(output_path, 'a') as fout:
        for res in results:
            for r in res:
                json.dump(r, fout)
                fout.write('\n')
                fout.flush()

    p.close()

    print(f"Done! Save to {output_path}")


if __name__ == '__main__':
    main()
