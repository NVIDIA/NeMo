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
This script can be used to sample phrases from yago_wiki intermediate file. Example:
   grammatica      Grammatica Latina (Leutschoviae, 1717)
   veteris;praecepta       Rhetorices veteris et novae praecepta (Lipsiae, 1717)
   institutiones;germanicae;hungaria;ortu  Institutiones linguac germanicae et slavicae in Hungaria ortu (Leutschoviae, 1718)
"""

from argparse import ArgumentParser
from collections import Counter

parser = ArgumentParser(description="Sample phrases")
parser.add_argument("--input_name", required=True, type=str, help="File with input data")
parser.add_argument("--output_phrases_name", required=True, type=str, help="File with output data, phrases part")
parser.add_argument("--output_paragraphs_name", required=True, type=str, help="File with output data, paragraphs part")
parser.add_argument(
    "--max_count",
    required=True,
    type=int,
    help="Maximum count after which we ignore lines that do not contain any new phrases",
)
parser.add_argument("--each_n_line", type=int, default=1, help="Take only each n-th line, default n=1")
args = parser.parse_args()

vocab = Counter()

out_phrases = open(args.output_phrases_name, "w", encoding="utf-8")
out_paragraph = open(args.output_paragraphs_name, "w", encoding="utf-8")

n = 0
with open(args.input_name, "r", encoding="utf-8") as f:
    for line in f:
        n += 1
        if n % args.each_n_line != 0:
            continue
        parts = line.strip().split("\t")
        phrase_str = parts[0]
        paragraph = " ".join(parts[1:])

        phrases = phrase_str.split(";")
        ok = False
        for phrase in phrases:
            if phrase not in vocab:
                vocab[phrase] = 0
            if vocab[phrase] < args.max_count:
                ok = True
                vocab[phrase] += 1
        if ok:
            out_phrases.write(phrase_str + "\n")
            out_paragraph.write(paragraph + "\n")

out_phrases.close()
out_paragraph.close()
