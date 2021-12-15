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
import json
import logging
import os
import random
from collections import Counter

from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels

random.seed(42)

"""
This script creates manifest file for speaker diarization inference purposes. 
Useful to get manifest when you have list of audio files and optionally rttm and uem files for evaluation

Note: make sure basename for each file is unique and rttm files also has the corresponding base name for mapping
"""


def write_file(name, lines, idx):
    with open(name, 'w') as fout:
        for i in idx:
            dic = lines[i]
            json.dump(dic, fout)
            fout.write('\n')
    logging.info("wrote", name)


def read_file(scp):
    scp = open(scp, 'r').readlines()
    return sorted(scp)


def main(wav_scp, text_scp=None, rttm_scp=None, uem_scp=None, ctm_scp=None, manifest_filepath=None):
    if os.path.exists(manifest_filepath):
        os.remove(manifest_filepath)

    wav_scp = read_file(wav_scp)
    len_wavs = len(wav_scp)

    if text_scp is not None:
        text_scp = read_file(text_scp)
        assert len(text_scp) == len_wavs
    else:
        text_scp = len(wav_scp) * [None]

    if rttm_scp is not None:
        rttm_scp = read_file(rttm_scp)
        assert len(rttm_scp) == len_wavs
    else:
        rttm_scp = len(wav_scp) * [None]

    if uem_scp is not None:
        uem_scp = read_file(uem_scp)
        assert len(uem_scp) == len_wavs
    else:
        uem_scp = len(wav_scp) * [None]

    if ctm_scp is not None:
        ctm_scp = read_file(ctm_scp)
    else:
        ctm_scp = len(wav_scp) * [None]

    lines = []
    for wav, text, rttm, uem, ctm in zip(wav_scp, text_scp, rttm_scp, uem_scp, ctm_scp):
        audio_line = wav.strip()
        if rttm is not None:
            rttm = rttm.strip()
            labels = rttm_to_labels(rttm)
            num_speakers = Counter([l.split()[-1] for l in labels]).keys().__len__()
        else:
            num_speakers = None

        if uem is not None:
            uem = uem.strip()

        if text is not None:
            text = open(text.strip()).readlines()[0].strip()
        else:
            text = "-"

        if ctm is not None:
            ctm = ctm.strip()

        meta = [
            {
                "audio_filepath": audio_line,
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": text,
                "num_speakers": num_speakers,
                "rttm_filepath": rttm,
                "uem_filepath": uem,
                "ctm_filepath": ctm,
            }
        ]
        lines.extend(meta)

    write_file(manifest_filepath, lines, range(len(lines)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths2audio_files", help="path to text file containing list of audio files", type=str, required=True
    )
    parser.add_argument("--paths2txt_files", help="path to text file containing list of transcription files", type=str)
    parser.add_argument("--paths2rttm_files", help="path to text file containing list of rttm files", type=str)
    parser.add_argument("--paths2uem_files", help="path to uem files", type=str)
    parser.add_argument("--paths2ctm_files", help="path to ctm files", type=str)
    parser.add_argument("--manifest_filepath", help="scp file name", type=str, required=True)

    args = parser.parse_args()

    main(
        args.paths2audio_files,
        args.paths2txt_files,
        args.paths2rttm_files,
        args.paths2uem_files,
        args.paths2ctm_files,
        args.manifest_filepath,
    )
