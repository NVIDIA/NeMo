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
This script creates NeMo manifests which can be used for training models on LJSpeech (split is taken from https://github.com/NVIDIA/tacotron2).
It optionally saves transcripts in .txt files (can be used for extracting durations via MFA library).
"""

import argparse
import json
import os

import sox
import wget
from nemo_text_processing.text_normalization.normalize import Normalizer

from nemo.collections.common.parts.preprocessing import parsers

parser = argparse.ArgumentParser()
parser.add_argument('--ljspeech_dir', required=True, type=str)
parser.add_argument('--normalizer_class', choices=["ENCharParser", "Normalizer"], default="Normalizer", type=str)
parser.add_argument('--whitelist_path', type=str)
parser.add_argument('--save_transcripts_in_txt', action='store_true', default=False)
parser.add_argument(
    '--manifest_text_var_is_normalized',
    action='store_true',
    default=False,
    help="If specified, the text in the manifest will contain normalized text. Otherwise, the text will contain the original text.",
)

args = parser.parse_args()


def main():
    filelist_base = 'https://raw.githubusercontent.com/NVIDIA/tacotron2/master/filelists'
    filelists = ['train', 'val', 'test']

    if args.normalizer_class == "ENCharParser":
        normalizer_call = parsers.make_parser(name='en')._normalize
    elif args.normalizer_class == "Normalizer":
        if args.whitelist_path is None:
            raise ValueError(
                "whitelist_path is required when using Normalizer. "
                "You can download the whitelist for LJSpeech from https://github.com/NVIDIA/NeMo/blob/main/nemo_text_processing/text_normalization/en/data/whitelist_lj_speech.tsv."
            )

        text_normalizer = Normalizer(lang="en", input_case="cased", whitelist=args.whitelist_path)
        text_normalizer_call_kwargs = {"punct_pre_process": True, "punct_post_process": True}

        normalizer_call = lambda x: text_normalizer.normalize(x, **text_normalizer_call_kwargs)
    else:
        raise ValueError("normalizer_class must be ENCharParser or Normalizer")

    for split in filelists:
        # Download file list if necessary
        filelist_path = os.path.join(args.ljspeech_dir, f"ljs_audio_text_{split}_filelist.txt")
        if not os.path.exists(filelist_path):
            wget.download(f"{filelist_base}/ljs_audio_text_{split}_filelist.txt", out=args.ljspeech_dir)

        manifest_target = os.path.join(args.ljspeech_dir, f"ljspeech_{split}.json")
        with open(manifest_target, 'w') as f_out:
            with open(filelist_path, 'r') as filelist:
                print(f"\nCreating {manifest_target}...")
                for line in filelist:
                    basename = line[6:16]

                    text = line[21:].strip()
                    norm_text = normalizer_call(text)

                    # Make sure corresponding wavfile exists
                    wav_path = os.path.join(args.ljspeech_dir, 'wavs', basename + '.wav')
                    assert os.path.exists(wav_path)

                    if args.save_transcripts_in_txt:
                        txt_path = os.path.join(args.ljspeech_dir, 'wavs', basename + '.txt')
                        with open(txt_path, 'w') as f_txt:
                            f_txt.write(norm_text)

                    # Write manifest entry
                    entry = {
                        'audio_filepath': wav_path,
                        'duration': sox.file_info.duration(wav_path),
                        'text': norm_text if args.manifest_text_var_is_normalized else text,
                        'normalized_text': norm_text,
                    }

                    f_out.write(json.dumps(entry) + '\n')


if __name__ == '__main__':
    main()
