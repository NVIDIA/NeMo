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
import json
import os

from nemo.utils import logging


"""
This file converts vad outputs to manifest file for speaker diarization purposes
present in vad output directory.
every vad line consists of start_time, end_time , speech/non-speech
vad_directory: path to directory of vad output files (we assume that vad filename matches with audio filename)
audio_directory: path to audio directory of audio files
manifest_file: output manifest file name
"""


def write_manifest(vad_directory, audio_directory, manifest_file):
    vad_files = glob.glob(vad_directory + "/*.txt")
    with open(manifest_file, 'w') as outfile:
        for vad_file in vad_files:
            f = open(vad_file, 'r')
            lines = f.readlines()
            audio_name = os.path.basename(vad_file).split('.')[0]
            for line in lines:
                vad_out = line.strip().split()
                start, dur, activity = float(vad_out[0]), float(vad_out[1]) - float(vad_out[0]), vad_out[2]
                if activity.lower() == 'speech':
                    audio_path = os.path.join(audio_directory, audio_name + '.wav')
                    meta = {"audio_filepath": audio_path, "offset": start, "duration": dur, "label": 'UNK'}
                    json.dump(meta, outfile)
                    outfile.write("\n")

            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vad_directory", help="path to vad directory", type=str, required=True)
    parser.add_argument(
        "--audio_directory",
        help="path to audio directory of audio files for which vad was computed",
        type=str,
        required=True,
    )
    parser.add_argument("--manifest_file", help="output manifest file name", type=str, required=True)

    args = parser.parse_args()
    vad_directory, audio_directory, manifest_file = (args.vad_directory, args.audio_directory, args.manifest_file)
    write_manifest(vad_directory, audio_directory, manifest_file)
    logging.info("wrote {} file from vad output files present in {}".format(manifest_file, vad_directory))
