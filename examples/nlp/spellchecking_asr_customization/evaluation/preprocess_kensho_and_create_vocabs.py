# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script can be used to preprocess Kensho corpus.

The input folder consists of subfolders like this
  ├── 0018ad922e541b415ae60e175160b976
  │   ├── 118.wav
  │   ├── 120.wav
  │   ├── 148.wav
  │   ├── 21.wav
  │   ├── 34.wav
  │   └── 3.wav


Transcription file looks like this:

wav_filename|wav_filesize|transcript
13aa6c0669adb5544a0d62beef677189/29.wav|165164|rather than something that we are more concerned by.
13aa6c0669adb5544a0d62beef677189/75.wav|198764|next year. If you think back to our Q2 results, we did announce at that point that
13aa6c0669adb5544a0d62beef677189/81.wav|161324|that will allow us to move to a higher level of production

Normalized file contains just normalized text (still keeping original case and punctuation)
Number of lines in it is the same as in original transcription file, except for header line.
"""

import argparse
import json
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_folder",
    required=True,
    type=str,
    help="Input folder in which each subfolder contains audios from the same talk",
)
parser.add_argument("--destination_folder", required=True, type=str, help="Destination folder with user vocabs")
parser.add_argument("--transcription_file", required=True, type=str, help="Original kensho file with transcriptions")
parser.add_argument("--normalized_file", required=True, type=str, help="Transcriptions after normalization")
parser.add_argument("--output_manifest", required=True, type=str, help="Output manifest in NeMo format")
parser.add_argument("--idf_file", required=True, type=str, help="File with idf of words and phrases")
parser.add_argument(
    "--min_idf_uppercase",
    type=float,
    default=5.0,
    help="Words with idf below that will be considered too frequent and won't be included in user vocabulary",
)
parser.add_argument(
    "--min_idf_lowercase",
    type=float,
    default=8.0,
    help="Words with idf below that will be considered too frequent and won't be included in user vocabulary",
)
parser.add_argument(
    "--min_len",
    required=True,
    type=int,
    help="Minimum number of characters in user phrase (including space), e.g. 6 symbols",
)

args = parser.parse_args()

EXCLUDE_PHRASES = {
    "certainly",
    "currently",
    "including",
    "briefly",
    "provided",
    "occur",
    "i'd",
    "i'll",
    "i've",
    "we'd",
    "we're",
    "we've",
    "you'd",
    "you'll",
    "you've",
    "you're",
    "they'd",
    "they're",
    "they've",
    "there'll",
    "consists",
    "concludes",
    "permits",
    "reflects",
    "discussed",
    "thank",
    "largely",
}


def parse_lines(csv_record, norm, idf):
    path, _, _ = csv_record.strip().split("|")
    user_id, audio_filename = path.split("/")

    text = norm.strip()
    text = re.sub(r"[\.\?\!]", r"\n", text)
    text = re.sub(r",", r" ,", text)
    text_parts = text.split("\n")

    custom_phrases = set()
    for part in text_parts:
        words = part.split()
        phrase_words = []
        for idx, w in enumerate(words):
            w_lower = w.casefold()
            if w == "" or w == "," or w == "--" or w_lower in EXCLUDE_PHRASES:
                if len(phrase_words) > 0:
                    custom_phrases.add(" ".join(phrase_words))
                phrase_words = []
                continue
            if w_lower != w and idx != 0:  # uppercase and not first word
                if w_lower not in idf or idf[w_lower] > args.min_idf_uppercase:
                    phrase_words.append(w)
                else:
                    if len(phrase_words) > 0:
                        custom_phrases.add(" ".join(phrase_words))
                    phrase_words = []
                continue
            if w_lower not in idf or idf[w_lower] > args.min_idf_lowercase and len(w) >= 4:  # lower-cased term
                if len(phrase_words) > 0:
                    custom_phrases.add(" ".join(phrase_words))
                custom_phrases.add(w)
                phrase_words = []
            else:
                if len(phrase_words) > 0:
                    custom_phrases.add(" ".join(phrase_words))
                phrase_words = []

    ref_text = norm.strip().replace(".", " ").replace("?", " ").replace("!", " ").replace("-", " ").replace(",", " ")
    ref_text = " ".join(ref_text.casefold().split())

    return user_id, audio_filename, ref_text, custom_phrases


if __name__ == "__main__":
    idf = {}
    with open(args.idf_file, "r", encoding="utf-8") as f:
        for line in f:
            phrase, score, freq = line.strip().split("\t")
            score = float(score)
            idf[phrase] = score

    csv_lines = []
    with open(args.transcription_file, "r", encoding="utf-8") as f:
        csv_lines = f.readlines()

    normalized_lines = []
    with open(args.normalized_file, "r", encoding="utf-8") as f:
        normalized_lines = f.readlines()

    if len(csv_lines) != 1 + len(normalized_lines):
        raise (
            IndexError,
            "number of lines in normalized and csv files do not match: "
            + str(len(normalized_lines))
            + " vs "
            + str(len(csv_lines)),
        )

    user2vocab = {}
    test_data = []
    for csv_record, norm in zip(csv_lines[1:], normalized_lines):
        user_id, audio_filename, ref_text, user_phrases = parse_lines(csv_record, norm, idf)
        if user_id not in user2vocab:
            user2vocab[user_id] = set()
        user2vocab[user_id] = user2vocab[user_id].union(user_phrases)
        record = {}
        record["text"] = ref_text
        record["audio_filepath"] = os.path.join(args.input_folder, user_id, audio_filename)
        record["doc_id"] = user_id
        test_data.append(record)

    # save manifest
    with open(args.output_manifest, "w", encoding="utf-8") as out:
        for d in test_data:
            line = json.dumps(d)
            out.write(line + "\n")

    # save user vocabs
    for user_id in user2vocab:
        with open(os.path.join(args.destination_folder, user_id + ".custom.txt"), "w", encoding="utf-8") as out:
            for phrase in user2vocab[user_id]:
                if len(phrase) >= args.min_len:
                    out.write(phrase + "\n")
