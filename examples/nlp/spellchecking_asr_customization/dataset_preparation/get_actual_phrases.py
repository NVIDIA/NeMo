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


from argparse import ArgumentParser
from collections import Counter


parser = ArgumentParser(
    description="Get list of all target phrases from Wikipedia corpus with their frequencies (counts one occurrence per paragraph)"
)
parser.add_argument("--input_file", required=True, type=str, help="Path to input file with phrases")
parser.add_argument("--output_file", type=str, required=True, help="Output file")

args = parser.parse_args()


vocab = Counter()
with open(args.input_file, "r", encoding="utf-8") as f:
    for line in f:
        text = line.strip().split("\t")[0]  # if line is tab-separated only first part is considered as text
        phrases = text.split(";")
        for phrase in phrases:
            vocab[phrase] += 1

with open(args.output_file, "w", encoding="utf-8") as out:
    for phrase, freq in vocab.most_common(10000000):
        out.write(phrase + "\t" + str(freq) + "\n")
