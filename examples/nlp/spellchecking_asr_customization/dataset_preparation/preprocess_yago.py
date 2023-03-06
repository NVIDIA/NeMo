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
This script can be used to preprocess Yago entities.
## Before running this script, download yagoTypes.tsv from
  https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/yago/downloads/
And run
  awk 'BEGIN {FS="\t"} {print $2}' < yagoTypes.tsv | sort -u
to get an input file for this script.
The input file looks like this:
    <Żywkowo,_Podlaskie_Voivodeship>
    <Żywkowo,_Warmian-Masurian_Voivodeship>
    <Żywocice>
    <ZYX>
    <Zyx_(cartoonist)>
    <ZyX_(company)>

The output file has two columns and looks like this:
    Żywkowo,_Podlaskie_Voivodeship         zywkowo_podlaskie_voivodeship
    Żywkowo,_Warmian-Masurian_Voivodeship  zywkowo_warmian-masurian_voivodeship
    Żywocice                               zywocice
    ZYX                                    zyx
    Zyx_(cartoonist)                       zyx_cartoonist
    ZyX_(company)                          zyx_company
"""

import re
from argparse import ArgumentParser
from nemo.collections.nlp.data.spellchecking_asr_customization.utils import replace_diacritics

parser = ArgumentParser(description="Clean YAGO entities")
parser.add_argument("--input_name", type=str, required=True, help="Input file")
parser.add_argument("--output_name", type=str, required=True, help="Output file")

args = parser.parse_args()

out = open(args.output_name, "w", encoding="utf-8")

with open(args.input_name, "r", encoding="utf-8") as inp:
    for line in inp:
        s = line.strip()
        s = s.replace("<", "").replace(">", "")  # delete <>
        key = s
        s = s.casefold()  # lowercase
        s = re.sub(r"\(.+\)", r"", s)  # delete brackets
        s = s.replace("_", " ")
        s = s.replace("/", ",")
        parts = s.split(",")
        for p in parts:
            sp = p.strip()
            if len(sp) < 3:
                continue
            if "." in sp:
                continue
            if re.match(r".*\d", sp):
                continue
            sp = replace_diacritics(sp)
            sp = "_".join(sp.split())
            if len(set(list(sp)) - set(list(" -'abcdefghijklmnopqrstuvwxyz"))) == 0:
                out.write(key + "\t" + sp + "\n")
            else:
                print(str(set(list(sp)) - set(list(" -'abcdefghijklmnopqrstuvwxyz"))))

out.close()
