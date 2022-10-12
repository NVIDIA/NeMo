# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script can be used to filter manifest with additional metrics.
"""

import json
from argparse import ArgumentParser
from tqdm.auto import tqdm

parser = ArgumentParser(description="Analyze errors in ASR predictions")
parser.add_argument("--input_manifest", required=True, type=str, help="Path to input manifest file")
parser.add_argument("--output_manifest", required=True, type=str, help="Path to output manifest file")
parser.add_argument("--max_wer", type=float, help="Maximum WER", default=30.0)
parser.add_argument("--max_cer", type=float, help="Maximum CER", default=20.0)
parser.add_argument("--max_start_cer", type=float, help="Maximum start CER", default=30.0)
parser.add_argument("--max_end_cer", type=float, help="Maximum end CER", default=30.0)
parser.add_argument("--max_len_ratio", type=float, help="Maximum length ratio", default=0.15)
args = parser.parse_args()

out = open(args.output_manifest, "w", encoding="utf-8")

duration = 0.0
with open(args.input_manifest, 'r', encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading manifest data"):
        line = line.replace("\n", "")
        data = json.loads(line)
        if (
            data["WER"] < args.max_wer
            and data["CER"] < args.max_cer
            and data["start_CER"] < args.max_start_cer
            and data["end_CER"] < args.max_end_cer
            and data["len_diff_ratio"] < args.max_len_ratio
        ):
            out.write(json.dumps(data) + "\n")
            duration += data["duration"]
out.close()
print("final duration=", duration)
