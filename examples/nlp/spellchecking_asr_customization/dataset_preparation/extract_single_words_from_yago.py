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
This script can be used to extract single words from Yago entities.
ATTENTION: Hyphenated words are split into parts.
The input file looks like this:
    Żywkowo,_Podlaskie_Voivodeship         zywkowo_podlaskie_voivodeship     
    Żywkowo,_Warmian-Masurian_Voivodeship  zywkowo_warmian-masurian_voivodeship   
    Żywocice                               zywocice
    ZYX                                    zyx
    Zyx_(cartoonist)                       zyx_cartoonist
    ZyX_(company)                          zyx_company

The output file has just single words split by characters:
    c a r t o o n i s t
    c o m p a n y
    m a s u r i a n
    p o d l a s k i e
    v o i v o d e s h i p
    w a r m i a n
    z y w k o w o
    z y w o c i c e
    z y x
"""

from argparse import ArgumentParser

parser = ArgumentParser(description="Extract single words from YAGO entities")
parser.add_argument("--input_name", type=str, required=True, help="Input file")
parser.add_argument("--output_name", type=str, required=True, help="Output file")

args = parser.parse_args()

vocab = set()

with open(args.input_name, "r", encoding="utf-8") as inp:
    for line in inp:
        orig, clean = line.strip().split("\t")
        words = clean.replace("-", " ").replace("_", " ").split(" ")
        for w in words:
            while w.startswith("'") and w.endswith("'"):
                w = w[1:-1]
            w = w.strip()
            if len(w) > 0:
                vocab.add(w)

with open(args.output_name, "w", encoding="utf-8") as out:
    for w in sorted(list(vocab)):
        out.write(" ".join(list(w)) + "\n")
