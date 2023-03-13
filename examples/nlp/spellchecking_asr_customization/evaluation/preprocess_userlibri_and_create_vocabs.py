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
This script can be used to preprocess UserLibri corpus.

The input folder consists of subfolders like this
  ├── audio_data
  |    ├── test-clean
  |    |    ├── speaker-1089-book-4217
  |    |    |    ├── 1089-134686-0000.wav
  |    |    |    |   ...
  |    |    |    ├── 1089-134686-0037.wav
  |    |    |    ├── 1089-134686.trans.txt      # 1089-134686-0001 STUFF IT INTO YOU HIS BELLY COUNSELLED HIM
  |    |    |    ├── 1089-134691-0000.wav
  |    |    |    |   ...
  |    |    |    ├── 1089-134691-0025.wav
  |    |    |    └── 1089-134691.trans.txt      
  |    |    |   ...
  |    |    └── speaker-908-book-574       
  |    ├── test-other
  |    └── metadata.tsv      # User ID, Split, Num Audio Examples, Average Words Per Example
  └── lm_data
       ├── 10136_lm_train.txt         # CALF'S HEAD A LA MAITRE D'HOTEL
       ├── 1041_lm_train.txt
       |   ...
       └── metadata.tsv    # Book ID, Num Text Examples, Average Words Per Example

Note that initial UserLibri audio files are in .flac format. Before running this script, please convert them to .wav:
       for i in */*.flac; do ffmpeg -i "$i" -ac 1 -ar 16000 "${i%.*}.wav"; done

Note, that transcription files do not require normalization.
Note, that in output NeMo manifest "doc_id" will be equal to book id, not user id.
Note, that we put data from test-clean and test-other into a single output manifest. If needed they can be separated afterwards by looking at "audio_path".

"""

import argparse
import json
import os
import re
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_folder", required=True, type=str, help="Input folder with UserLibri corpus, see structure above."
)
parser.add_argument("--destination_folder", required=True, type=str, help="Destination folder with user vocabs")
parser.add_argument("--output_manifest", required=True, type=str, help="Output manifest in NeMo format")
parser.add_argument("--idf_file", required=True, type=str, help="File with idf of words and phrases")
parser.add_argument(
    "--min_idf",
    type=float,
    default=5.0,
    help="Words with idf below that will be considered too frequent and won't be included in user vocabulary",
)
parser.add_argument(
    "--min_len",
    required=True,
    type=int,
    help="Minimum number of characters in user phrase (including space), e.g. 6 symbols",
)
parser.add_argument(
    "--use_lm_for_vocab",
    action="store_true",
    help="Whether to additionally extract vocabularies from lm files. If false (default) only transcriptions will be used",
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
    "you're",
    "you've",
    "they'd",
    "they're",
    "they've",
    "there'll",
    "that'll",
    "consists",
    "concludes",
    "permits",
    "reflects",
    "discussed",
    "thank",
    "largely",
    "whomsoever",
    "whosoever",
    "whereon",
    "anyhow",
    "unsaid",
}


def extract_custom_phrases(text, idf):
    text = re.sub(r"[\.\?\!\,\:]", r" ", text)
    text = re.sub(r"--", r" ", text)
    words = text.split()

    custom_phrases = set()
    for w in words:
        if w == "" or w in EXCLUDE_PHRASES:
            continue
        if w.startswith("'") or w.endswith("'"):
            continue
        if "'" in w:
            continue
        if (
            w.endswith("ed")
            or w.endswith("ing")
            or w.endswith("'d")
            or w.endswith("in")
            or w.endswith("'s")
            or w.endswith("es")
            or w.endswith("ly")
        ):
            continue

        if re.match(r"^[a-z\-']+$", w):  # check that a string only contains letters, apostrophe or hyphen
            if w not in idf or idf[w] > args.min_idf:
                custom_phrases.add(w)
    return custom_phrases


if __name__ == "__main__":
    idf = {}
    with open(args.idf_file, "r", encoding="utf-8") as f:
        for line in f:
            phrase, score, freq = line.strip().split("\t")
            score = float(score)
            idf[phrase] = score

    test_data = []
    book2vocab = {}

    for split in ["test-clean", "test-other"]:
        for folder_name in os.listdir(os.path.join(args.input_folder, "audio_data", split)):
            print(folder_name)
            _, user_id, _, book_id = folder_name.split("-")
            for name in os.listdir(os.path.join(args.input_folder, "audio_data", split, folder_name)):
                if not name.endswith(".trans.txt"):
                    continue
                with open(
                    os.path.join(args.input_folder, "audio_data", split, folder_name, name), "r", encoding="utf-8"
                ) as f:
                    for line in f:
                        text = line.strip().casefold()
                        words = text.split(" ")
                        file_id = words[0]
                        words = words[1:]
                        record = {}
                        record["text"] = " ".join(words)
                        record["audio_filepath"] = os.path.join(
                            args.input_folder, "audio_data", split, folder_name, file_id + ".wav"
                        )
                        record["doc_id"] = book_id
                        test_data.append(record)

                        custom_phrases = extract_custom_phrases(" ".join(words), idf)
                        if book_id not in book2vocab:
                            book2vocab[book_id] = set()
                        book2vocab[book_id] = book2vocab[book_id].union(custom_phrases)

    # save manifest
    with open(args.output_manifest, "w", encoding="utf-8") as out:
        for d in test_data:
            line = json.dumps(d)
            out.write(line + "\n")

    if args.use_lm_for_vocab:
        for name in os.listdir(os.path.join(args.input_folder, "lm_data")):
            if not name.endswith("_lm_train.txt"):
                continue
            book_id, _, _ = name.split("_")

            with open(os.path.join(args.input_folder, "lm_data", name), "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip().casefold()
                    custom_phrases = extract_custom_phrases(text, idf)
                    if book_id not in book2vocab:
                        book2vocab[book_id] = set()
                    book2vocab[book_id] = book2vocab[book_id].union(custom_phrases)

    # save book vocabs
    for book_id in book2vocab:
        with open(os.path.join(args.destination_folder, book_id + ".custom.txt"), "w", encoding="utf-8") as out:
            for phrase in book2vocab[book_id]:
                if len(phrase) >= args.min_len:
                    out.write(phrase + "\n")
