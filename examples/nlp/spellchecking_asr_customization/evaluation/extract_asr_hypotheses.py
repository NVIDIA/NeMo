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


import json
from argparse import ArgumentParser
from collections import defaultdict

from tqdm.auto import tqdm

parser = ArgumentParser(description="Extract shorter ASR hypotheses for spellchecker customization")
parser.add_argument("--manifest", required=True, type=str, help='Path to manifest file')
parser.add_argument("--folder", required=True, type=str, help='Path to output folder')
args = parser.parse_args()


def read_manifest(path):
    manifest = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest


def save_hypotheses(hypotheses, doc_id):
    with open(args.folder + "/" + doc_id + ".txt", "w", encoding="utf-8") as out:
        for short, full in hypotheses:
            out.write(short + "\t" + full + "\n")


test_data = read_manifest(args.manifest)

# extract just the text corpus from the manifest
pred_texts = [data['pred_text'] for data in test_data]
audio_filepaths = [data['audio_filepath'] for data in test_data]
doc_ids = []

for data in test_data:
    if "doc_id" in data:
        doc_ids.append(data["doc_id"])
    else:  # fix for Spoken Wikipedia format
        path = data["audio_filepath"]
        # example of path: ...clips/197_0000.wav   #doc_id=197
        path_parts = path.split("/")
        path_parts2 = path_parts[-1].split("_")
        doc_id = path_parts2[-2]
        doc_ids.append(doc_id)

doc2hypotheses = defaultdict(list)
for sent, path, doc_id in zip(pred_texts, audio_filepaths, doc_ids):
    words = sent.split()
    for i in range(0, len(words), 2):
        short_sent = " ".join(words[i : i + 10])
        if len(short_sent) > 8:
            doc2hypotheses[doc_id].append((short_sent, sent))

for doc_id in doc2hypotheses:
    save_hypotheses(doc2hypotheses[doc_id], doc_id)
