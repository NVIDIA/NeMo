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
import json
import os
import random
import shutil

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest

random.seed(42)

"""
This script creates a manifest file containing word alignments.

The alignments are obtained from: https://github.com/CorentinJ/librispeech-alignments
"""

# get librispeech examples without alignments for the desired dataset
def get_unaligned_examples(unaligned_path, dataset):
    with open(unaligned_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        i = 0
        skip_files = []
        while i < len(lines):
            l = lines[i].strip('\n')
            if l[0] == '#':
                unaligned_dataset = l.split(" ")[1]
            elif unaligned_dataset == dataset:
                unaligned_file = l.split(" ")[0]
                skip_files.append(unaligned_file)
            i += 1
    return skip_files


def main():
    input_manifest_filepath = args.input_manifest_filepath
    base_alignment_path = args.base_alignment_path
    output_path = args.output_path
    dataset = args.dataset

    manifest = read_manifest(input_manifest_filepath)
    target_manifest = []
    unaligned_path = os.path.join(base_alignment_path, "unaligned.txt")
    unaligned = get_unaligned_examples(unaligned_path, dataset)
    num_unaligned = len(unaligned)

    # separate indices to manage source/destination manifest to handle missing alignments
    src_i = 0
    target_i = 0
    while src_i < len(manifest):
        file = manifest[src_i]
        fn = file['audio_filepath'].split('/')[-1]
        speaker_id = fn.split('-')[0]
        book_id = fn.split('-')[1]

        book_dir = os.path.join(base_alignment_path, "LibriSpeech", dataset, speaker_id, book_id)
        alignment_fpath = os.path.join(book_dir, f"{speaker_id}-{book_id}.alignment.txt")

        # Parse each utterance present in the file
        alignment_file = open(alignment_fpath, "r")
        for line in alignment_file:

            # filter out unaligned examples
            file = manifest[src_i]
            fn = file['audio_filepath'].split('/')[-1]
            line_id = fn.split('.')[0]
            while line_id in unaligned:
                src_i += 1
                file = manifest[src_i]
                fn = file['audio_filepath'].split('/')[-1]
                line_id = fn.split('.')[0]

            # from https://github.com/CorentinJ/librispeech-alignments/blob/master/parser_example.py
            # Retrieve the utterance id, the words as a list and the end_times as a list
            utterance_id, words, end_times = line.strip().split(' ')
            if utterance_id != line_id:
                raise Exception("Mismatch between source and target utterance id")
            words = words.replace('\"', '').lower().split(',')
            end_times = [float(e) for e in end_times.replace('\"', '').split(',')]

            # get speaker ID
            fn = file['audio_filepath'].split('/')[-1]
            speaker_id = fn.split('-')[0]

            # build target manifest entry
            target_manifest.append({})
            target_manifest[target_i]['audio_filepath'] = file['audio_filepath']
            target_manifest[target_i]['duration'] = file['duration']
            target_manifest[target_i]['text'] = file['text']
            target_manifest[target_i]['words'] = words
            target_manifest[target_i]['alignments'] = end_times
            target_manifest[target_i]['speaker_id'] = speaker_id

            src_i += 1
            target_i += 1

        alignment_file.close()
    write_manifest(output_path, target_manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LibriSpeech Alignment Manifest Creator")
    parser.add_argument("--input_manifest_filepath", help="path to input manifest file", type=str, required=True)
    parser.add_argument("--base_alignment_path", help="path to librispeech alignment dataset", type=str, required=True)
    parser.add_argument(
        "--dataset", help="which test/dev/training set to create a manifest for", type=str, required=True
    )
    parser.add_argument("--output_path", help="path to output manifest file", type=str, required=True)
    args = parser.parse_args()

    main()
