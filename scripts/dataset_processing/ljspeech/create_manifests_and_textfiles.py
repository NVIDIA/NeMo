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
Creates the manifest and .txt transcript files for an LJSpeech split.
The manifest will be used for training, and the .txt files are for the MFA library to find.
"""

import argparse
import json
import os

import sox
import wget

from nemo.collections.asr.parts import parsers

parser = argparse.ArgumentParser()
parser.add_argument('--ljspeech_base', required=True, default=None, type=str)
args = parser.parse_args()


def main():
    filelist_base = 'https://raw.githubusercontent.com/NVIDIA/tacotron2/master/filelists/'
    filelists = ['train', 'val', 'test']

    # NeMo parser for text normalization
    text_parser = parsers.make_parser(name='en')

    for split in filelists:
        # Download file list if necessary
        filelist_path = os.path.join(args.ljspeech_base, f"ljs_audio_text_{split}_filelist.txt")
        if not os.path.exists(filelist_path):
            wget.download(f"{filelist_base}/ljs_audio_text_{split}_filelist.txt", out=args.ljspeech_base)

        manifest_target = os.path.join(args.ljspeech_base, f"ljspeech_{split}.json")
        with open(manifest_target, 'w') as f_out:
            with open(filelist_path, 'r') as filelist:
                print(f"\nCreating {manifest_target}...")
                for line in filelist:
                    basename = line[6:16]
                    text = text_parser._normalize(line[21:].strip())

                    # Make sure corresponding wavfile exists and write .txt transcript
                    wav_path = os.path.join(args.ljspeech_base, 'wavs/', basename + '.wav')
                    assert os.path.exists(wav_path)
                    txt_path = os.path.join(args.ljspeech_base, 'wavs/', basename + '.txt')
                    with open(txt_path, 'w') as f_txt:
                        f_txt.write(text)

                    # Write manifest entry
                    entry = {
                        'audio_filepath': wav_path,
                        'duration': sox.file_info.duration(wav_path),
                        'text': text,
                    }
                    f_out.write(json.dumps(entry) + '\n')


if __name__ == '__main__':
    main()
