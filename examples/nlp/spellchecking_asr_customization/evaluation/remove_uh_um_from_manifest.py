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

import argparse
import json

from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_manifest", required=True, type=str, help="Manifest with transcription before correction")
parser.add_argument("--output_manifest", required=True, type=str, help="Manifest with transcription after correction")

args = parser.parse_args()


def read_manifest(path):
    manifest = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest


test_data = read_manifest(args.input_manifest)

for i in range(len(test_data)):
    test_data[i]["pred_text"] = test_data[i]["pred_text"].replace(" um ", " ").replace(" uh ", " ")
    if test_data[i]["pred_text"].startswith("um ") or test_data[i]["pred_text"].startswith("uh "):
        test_data[i]["pred_text"] = test_data[i]["pred_text"][3:]

with open(args.output_manifest, "w", encoding="utf-8") as out:
    for d in test_data:
        line = json.dumps(d)
        out.write(line + "\n")
