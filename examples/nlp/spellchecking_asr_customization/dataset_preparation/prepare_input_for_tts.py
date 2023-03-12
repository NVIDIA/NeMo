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
This script can be used to prepare input for tts from Yago entities and their g2p translations.

The yago_input_name file looks like this:
    Żywkowo,_Podlaskie_Voivodeship         zywkowo_podlaskie_voivodeship
    Żywkowo,_Warmian-Masurian_Voivodeship  zywkowo_warmian-masurian_voivodeship
    Żywocice                               zywocice
    ZYX                                    zyx
    Zyx_(cartoonist)                       zyx_cartoonist
    ZyX_(company)                          zyx_company

The phonematic input file looks like this (only columns 2 and 3 matter):
    Z AY1 W EH2 B  \t  z y w e b  \t  Z AY1 W EH2 B  \t  Z AY1 W EH2 B  \t  PLAIN PLAIN PLAIN PLAIN PLAIN

The output file has the following format (space is also a phoneme):
    aadityana       AA0,AA2,D,AH0,T,Y,AE1,N,AH0
    aadivaram aadavallaku selavu    AA2,D,IH1,V,ER0,AE2,M, ,AA2,AA0,D,AH0,V,AA1,L,AA1,K,UW2, ,S,EH1,L,AH0,V,UW0
    aa divasam      EY1,EY1, ,D,IH0,V,AH0,S,AA1,M
    aadi velli      AA1,D,IY0, ,V,EH1,L,IY0
"""

from argparse import ArgumentParser

parser = ArgumentParser(description="Prepare input for TTS")
parser.add_argument("--yago_input_name", type=str, required=True, help="Input file with yago entities")
parser.add_argument("--phonematic_name", type=str, required=True, help="Input file with tagger g2p output")
parser.add_argument("--output_name", type=str, required=True, help="Output file")

args = parser.parse_args()

vocab = {}

with open(args.phonematic_name, "r", encoding="utf-8") as inp:
    for line in inp:
        parts = line.strip().split("\t")
        pred = parts[2]
        pred = pred.replace("<DELETE>", "").replace("_", " ")
        pred = " ".join(pred.split())
        if pred == "":
            continue
        inp = parts[1]
        vocab[parts[1]] = pred

out = open(args.output_name, "w", encoding="utf-8")

with open(args.yago_input_name, "r", encoding="utf-8") as inp:
    for line in inp:
        orig, clean = line.strip().split("\t")
        s = " ".join(clean.replace("_", " ").replace("-", " ").split())
        parts = s.split(" ")
        res = []
        ok = True
        for p in parts:
            k = " ".join(list(p))
            if k not in vocab:
                print("not found: " + k)
                ok = False
                continue
            v = vocab[k]
            res.extend(v.split())
            res.append(" ")
        if len(res) > 26:
            print("too long: ", s, res)
            ok = False
        if ok:
            res = res[:-1]
            out.write(s + "\t" + ",".join(res) + "\n")

out.close()
