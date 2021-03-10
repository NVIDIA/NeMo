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

from nemo.collections.asr.parts.speaker_utils import write_rttm2manifest
from nemo.utils import logging

"""
This file converts vad outputs to manifest file for speaker diarization purposes
present in vad output directory.
every vad line consists of start_time, end_time , speech/non-speech
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths2rttm_files", help="path to vad output rttm-like files. Could be a list or a text file", required=True
    )
    parser.add_argument(
        "--paths2audio_files",
        help="path to audio files that vad was computed. Could be a list or a text file",
        required=True,
    )
    parser.add_argument("--manifest_file", help="output manifest file name", type=str, required=True)
    args = parser.parse_args()

    write_rttm2manifest(args.paths2audio_files, args.paths2rttm_files, args.manifest_file)
    logging.info("wrote {} file from vad output files present in {}".format(args.manifest_file, args.paths2rttm_files))
