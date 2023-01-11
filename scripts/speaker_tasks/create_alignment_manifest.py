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
import math
import json

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_ctm, write_manifest
from nemo.utils import logging

random.seed(42)


def get_unaligned_examples(unaligned_path, dataset):
    """
    Get librispeech examples without alignments for the desired dataset in
    order to filter them out (as they cannot be used for data simulation))

    Args:
        unaligned_path (str): Path to the file containing unaligned examples
        dataset (str): LibriSpeech data split being used

    Returns:
        skip_files (list): Unaligned file names to skip
    """
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


def create_new_ctm_entry(session_name, speaker_id, wordlist, alignments, output_precision=3):
    """
    Create new CTM entry (to write to output ctm file)

    Args:
        session_name (str): Current session name.
        speaker_id (int): LibriSpeech speaker ID for the current entry.
        wordlist (list): List of words
        alignments (list): List of alignments
        output_precision (int): Precision for CTM outputs
    Returns:
        arr (list): List of ctm entries
    """
    arr = []
    for i in range(len(wordlist)):
        word = wordlist[i]
        if word != "":
            # note that using the current alignments the first word is always empty, so there is no error from indexing the array with i-1
            align1 = float(round(alignments[i - 1], output_precision))
            align2 = float(round(alignments[i] - alignments[i - 1], output_precision,))
            text = f"{session_name} {speaker_id} {align1} {align2} {word} 0\n"
            arr.append((align1, text))
    return arr


def create_ctm_alignments(input_manifest_filepath, base_alignment_path, ctm_directory, dataset):
    """
    Create new CTM alignments using input LibriSpeech word alignments. 

    Args:
        input_manifest_filepath (str): Path to the input LibriSpeech manifest file
        base_alignment_path (str): Path to the base directory containing the LibriSpeech word alignments
        ctm_directory (str): Directory to write the CTM files to
        dataset (str): Which split of the LibriSpeech dataset is being used
    """
    manifest = read_manifest(input_manifest_filepath)
    unaligned_path = os.path.join(base_alignment_path, "unaligned.txt")
    dataset = dataset.replace("_", "-")
    unaligned = get_unaligned_examples(unaligned_path, dataset)

    # delete output directory if it exists or throw warning
    if os.path.isdir(ctm_directory):
        shutil.rmtree(ctm_directory)
    os.mkdir(ctm_directory)

    idx = 0
    while idx < len(manifest):
        f = manifest[idx]
        # get speaker_id
        fn = f['audio_filepath'].split('/')[-1]
        speaker_id = fn.split('-')[0]
        book_id = fn.split('-')[1]

        book_dir = os.path.join(base_alignment_path, "LibriSpeech", dataset, speaker_id, book_id)
        alignment_fpath = os.path.join(book_dir, f"{speaker_id}-{book_id}.alignment.txt")

        # Parse each utterance present in the file
        alignment_file = open(alignment_fpath, "r")
        for line in alignment_file:

            # filter out unaligned examples
            f = manifest[idx]
            fn = f['audio_filepath'].split('/')[-1]
            line_id = fn.split('.')[0]
            while line_id in unaligned:
                idx += 1
                f = manifest[idx]
                fn = f['audio_filepath'].split('/')[-1]
                line_id = fn.split('.')[0]

            # from https://github.com/CorentinJ/librispeech-alignments/blob/master/parser_example.py
            # Retrieve the utterance id, the words as a list and the end_times as a list
            utterance_id, words, end_times = line.strip().split(' ')
            if utterance_id != line_id:
                raise Exception("Mismatch between source and target utterance id")
            words = words.replace('\"', '').lower().split(',')
            end_times = [float(e) for e in end_times.replace('\"', '').split(',')]

            ctm_list = create_new_ctm_entry(line_id, speaker_id, words, end_times)
            write_ctm(os.path.join(ctm_directory, line_id + '.ctm'), ctm_list)
            idx += 1
        alignment_file.close()

def breakdown_manifest(target_manifest, breakdown_thres=15):
    """
    Breakdown manifest into smaller chunks if the number of words exceeds the threshold.

    Args:
        target_manifest (list): 
            List of dictionaries containing the manifest information.
        breakdown_thres (int):
            Threshold for the number of words in a line. If the number of words exceeds this threshold, 
            the json line will be broken down into multiple lines.

    Returns:
        output_manifest_list (list): 
            List of dictionaries containing the manifest information.
    """
    output_manifest_list = []
    for json_dict in target_manifest:
        audio_filepath = json_dict['audio_filepath']
        duration = json_dict['duration']
        speaker_id = json_dict['speaker_id']
        words = json_dict['words']
        alignments = json_dict['alignments']
        assert len(words) == len(alignments), f"Mismatch between words {len(words)} and alignments {len(alignments)} for {audio_filepath}"
        last_alignment = alignments[0]
        if len(words) > breakdown_thres:
            # split the line into multiple lines
            num_lines = math.ceil(len(words) / breakdown_thres)
            for i in range(num_lines):
                start = i * breakdown_thres
                end = (i + 1) * breakdown_thres
                if end > len(words):
                    end = len(words)
                if i > 0:
                    add_offset_sil = [""]
                    add_offset_stamp = [last_alignment]
                else:
                    add_offset_sil = []
                    add_offset_stamp = []

                new_json_dict = {
                    'audio_filepath': audio_filepath,
                    'duration': duration,
                    'speaker_id': speaker_id,
                    'words': add_offset_sil + words[start:end],
                    'alignments': add_offset_stamp + alignments[start:end],
                }
                last_alignment = alignments[end - 1]
                output_manifest_list.append(new_json_dict)
        else:
            output_manifest_list.append(json_dict)
    return output_manifest_list

def create_manifest_with_alignments(
    input_manifest_filepath, ctm_directory, output_manifest_filepath, data_format_style, silence_dur_threshold=0.1, output_precision=3, breakdown_thres=15
):
    """
    Create new manifest file with word alignments using CTM files

    Args:
        input_manifest_filepath (str): Path to the input manifest file
        ctm_directory (str): Directory to read the CTM files from
        output_manifest_filepath (str): Path to the output manifest file containing word alignments
        precision (int): How many decimal places to keep in the manifest file
    """
    manifest = read_manifest(input_manifest_filepath)

    target_manifest = []
    src_i = 0
    tgt_i = 0
    while src_i < len(manifest):
        f = manifest[src_i]
        fn = f['audio_filepath'].split('/')[-1]
        filename = fn.split('.')[0]  # assuming that there is only one period in the input filenames

        if "voxceleb" in data_format_style:
            fn_split = f['audio_filepath'].split('/')
            filename = fn_split[-3] + '-' + fn_split[-2] + '-' + fn_split[-1].split('.')[0]
            ctm_filepath = os.path.join(ctm_directory, filename + '.ctm')
        else:
            ctm_filepath = os.path.join(ctm_directory, filename + '.ctm')

        if not os.path.isfile(ctm_filepath):
            logging.info(f"Skipping {filename}.wav as there is no corresponding CTM file")
            src_i += 1
            continue

        with open(ctm_filepath, 'r') as ctm_file:
            lines = ctm_file.readlines()

        # One-word samples should be filtered out.
        if len(lines) <= 1:
            src_i += 1
            continue

        words = []
        end_times = []
        i = 0
        prev_end = 0
        for i in range(len(lines)):
            ctm = lines[i].split(' ')
            speaker_id = ctm[1]
            start = float(ctm[2])
            end = float(ctm[2]) + float(ctm[3])
            start = round(start, output_precision)
            end = round(end, output_precision)
            interval = start - prev_end 

            if (i == 0 and interval > 0) or (i > 0 and interval > silence_dur_threshold):
                words.append("")
                end_times.append(start)
            elif i > 0 and interval <= silence_dur_threshold:
                end_times[-1] = start

            words.append(ctm[4])
            end_times.append(end)

            i += 1
            prev_end = end

        # append last end
        if f['duration'] > prev_end:
            words.append("")
            end_times.append(f['duration'])

        # build target manifest entry
        target_manifest.append({})
        target_manifest[tgt_i]['audio_filepath'] = f['audio_filepath']
        target_manifest[tgt_i]['duration'] = f['duration']
        target_manifest[tgt_i]['text'] = f['text']
        target_manifest[tgt_i]['words'] = words
        target_manifest[tgt_i]['alignments'] = end_times
        target_manifest[tgt_i]['speaker_id'] = speaker_id

        src_i += 1
        tgt_i += 1

    target_manifest = breakdown_manifest(target_manifest, breakdown_thres)
    logging.info(f"Writing output manifest file to {output_manifest_filepath}")
    write_manifest(output_manifest_filepath, target_manifest)


def main():
    """
    Create a combined manifest file including word alignments and speaker IDs
    """
    input_manifest_filepath = args.input_manifest_filepath
    base_alignment_path = args.base_alignment_path
    output_manifest_filepath = args.output_manifest_filepath
    ctm_directory = args.ctm_directory
    dataset = args.dataset
    use_ctm = args.use_ctm
    output_precision = args.output_precision

    # args.base_alignment_path is containing the ctm files
    if use_ctm:
        ctm_directory = args.base_alignment_path
    # args.base_alignment_path is containing *.lab style alignments for the dataset
    else:
        create_ctm_alignments(input_manifest_filepath, base_alignment_path, ctm_directory, dataset)

    create_manifest_with_alignments(
        input_manifest_filepath, 
        ctm_directory, 
        output_manifest_filepath, 
        data_format_style=args.data_format_style,
        silence_dur_threshold=args.silence_dur_threshold,
        output_precision=output_precision, 
        breakdown_thres=args.breakdown_thres,
    )


if __name__ == "__main__":
    """
    This script creates a manifest file to be used for generating synthetic
    multispeaker audio sessions. The script takes in the default manifest file
    for a LibriSpeech dataset and corresponding word alignments and produces
    a combined manifest file that contains word alignments and speaker IDs
    per example. It can also be used to produce a manifest file for a different
    dataset if alignments are passed in CTM files.

    The alignments are obtained from: https://github.com/CorentinJ/librispeech-alignments

    Args:
        input_manifest_filepath (str): Path to input manifest file
        base_alignment_path (str): Path to the base directory for the LibriSpeech alignment dataset 
                                   (specifically to the LibriSpeech-Alignments directory containing 
                                   both the LibriSpeech folder as well as the unaligned.txt file) 
                                   or to a directory containing the requisite CTM files
        output_manifest_filepath (str): Path to output manifest file
        ctm_directory (str): Path to output CTM directory (only used for LibriSpeech)
        dataset (str): Which dataset split to create a combined manifest file for
        use_ctm (bool): If true, base_alignment_path points to a directory containing ctm files
    """
    parser = argparse.ArgumentParser(description="LibriSpeech Alignment Manifest Creator")
    parser.add_argument("--input_manifest_filepath", help="path to input manifest file", type=str, required=True)
    parser.add_argument("--base_alignment_path", help="path to alignments (LibriSpeech)", type=str, required=False)
    parser.add_argument("--output_manifest_filepath", help="path to output manifest file", type=str, required=True)
    parser.add_argument(
        "--ctm_directory",
        help="path to output ctm directory for LibriSpeech (or to input CTM directory)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        help="which test/dev/training set to create a manifest for (only used for LibriSpeech)",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--use_ctm",
        help="if true, base_alignment_path points to a directory containing ctm files",
        action='store_true',
        required=False,
    )
    parser.add_argument(
        "--data_format_style",
        help="Use specific format for speaker IDs and utterance IDs. e.g. 'voxceleb', 'librispeech', 'swbd'",
        default=False,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--output_precision", help="precision for output alignments", type=int, required=False, default=3
    )
    parser.add_argument(
        "--breakdown_thres", help="threshold for breaking-down the source utterances", type=int, required=False, default=15
    )
    parser.add_argument(
        "--silence_dur_threshold", help="threshold for inserting silence", type=float, required=False, default=0.1
    )
    args = parser.parse_args()

    main()
