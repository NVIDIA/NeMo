# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
import random
import shutil

from nemo.collections.asr.parts.preprocessing.diarization import LibriSpeechGenerator

random.seed(42)

"""
This script creates a synthetic diarization dataset using the LibriSpeech dataset.
"""

def main():
    input_manifest_filepath = args.input_manifest_filepath
    output_dir = args.output_dir
    num_sessions = args.num_sessions
    session_length = args.session_length
    output_filename = args.output_filename

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    lg = LibriSpeechGenerator(manifest_path=input_manifest_filepath,output_dir=output_dir)

    for i in range(num_sessions):
        #update output_filename, session_length
        lg.set_session_length(session_length)
        lg.set_output_filename(output_filename+'_{}'.format(i))
        lg.generate_session()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LibriSpeech Synthetic Diarization Generator")
    parser.add_argument("--input_manifest_filepath", help="path to input manifest file", type=str, required=True)
    parser.add_argument("--output_dir", help="path to output directory", type=str, required=True)
    parser.add_argument("--num_sessions", help="number of diarization sessions", type=str, default=1)
    parser.add_argument("--session_length", help="length of each diarization session (seconds)", type=int, default=60)
    parser.add_argument("--output_filename", help="filename for wav and rttm files", type=str, default='diarization_session')
    args = parser.parse_args()

    main()
