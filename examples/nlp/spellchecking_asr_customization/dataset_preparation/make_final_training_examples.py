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
This script makes final training examples in format that model expects them to be.

It takes as input two files: positive and negative.
It does the following:
    - substitutes misspells to the text, recalculates positions of targets
    - makes sampling of whole examples if needed
    - writes output in expected format


Negative example consists of 2 columns:
    original text fragment
    10 candidates

Positive example consists of 5 columns:
    original text fragment
    10 candidates
    targets ids
    target spans
    target misspells
Example:
    ninjas knuckle up and three ninjas kick back as	
    *beckjay;&jean-louis bonenfant;knuckle;*layback spin;*ninjatown;&antecedent;*grabocka;&ashrita;*licinius;ninjas
    3 10 10
    7 14;0 6;28 34
    knockle;ninjs;ninjas

"""

import random
import re
from argparse import ArgumentParser


parser = ArgumentParser(description="Preparation of final training examples")

# three ninjas three ninjas knuckle up and three ninjas	*panda's;knuckle;*therrien;&william flynt nichols;&cansahcab;*wpk;&yenmanagandla;*neniae;*wpb;ninjas	2 10 10 10	26 33;19 25;6 12;47 53	knuckle;nines;ninjas;dinges
parser.add_argument("--positive_file", required=True, type=str, help="Input file with positive examples")

# merchant's house ceased	*a house;*windhausen;*oase;*seasider;*gladhouse;*haase;*hoseyni;*hosen;*gunhouse;*caseyi
parser.add_argument("--negative_file", required=True, type=str, help="Input file with negative examples")

parser.add_argument(
    "--fraction_of_negatives",
    type=float,
    default=0.5,
    help="Desired fraction of negative examples, from 0 to 1, default 0.5",
)


parser.add_argument("--output_file", required=True, type=str, help="Output file")
args = parser.parse_args()


def main() -> None:
    input_positive = open(args.positive_file, "r", encoding="utf-8")
    input_negative = open(args.negative_file, "r", encoding="utf-8")
    output = open(args.output_file, "w", encoding="utf-8")
    while True:
        if random.uniform(0, 1) > args.fraction_of_negatives:
            f = input_positive
        else:
            f = input_negative
        line = f.readline()
        if not line:
            break
        parts = line.strip().split("\t")
        if len(parts) == 2:  # negative example
            text, candidate_str = parts
            text = " ".join(list(text.replace(" ", "_")))
            if text != re.sub("[^ _'\-aiuenrbomkygwthszdcjfvplxq]", " ", text):
                print("BAD text: ", line)
                continue
            if candidate_str != re.sub("[^ ';*#&\-aiuenrbomkygwthszdcjfvplxq]", " ", candidate_str):
                print("BAD candidate_str: ", line)
                continue
            candidates = []
            for cand in candidate_str.split(";"):
                candidate = " ".join(list(cand[1:].replace(" ", "_")))
                candidates.append(candidate)
            output.write(text + "\t" + ";".join(candidates) + "\t0\t\n")
        elif len(parts) == 5:  # positive example
            text, candidate_str, target_str, span_str, misspell_str = parts
            if text != re.sub("[^ '\-aiuenrbomkygwthszdcjfvplxq]", " ", text):
                print("BAD text: ", line)
                continue
            if candidate_str != re.sub("[^ ';*#&\-aiuenrbomkygwthszdcjfvplxq]", " ", candidate_str):
                print("BAD candidate_str: ", line)
                continue
            if misspell_str != re.sub("[^ ';\-aiuenrbomkygwthszdcjfvplxq]", " ", misspell_str):
                print("BAD misspell_str: ", line)
                continue
            spans = []
            for sp in span_str.split(";"):
                begin, end = sp.split(" ")
                spans.append([int(begin), int(end)])
            misspells = misspell_str.split(";")
            if len(misspells) != len(spans):
                raise (IndexError, "mismatch in length of spans and misspells: " + line)
            for i in range(len(spans)):
                # 0 if same length, negative if misspell is shorter, positive if misspell is longer
                begin = spans[i][0]
                end = spans[i][1]
                len_diff = len(misspells[i]) - (end - begin)
                text = text[:begin] + misspells[i] + text[end:]
                if len_diff != 0:
                    for j in range(len(spans)):  # update all positions further than begin
                        if spans[j][0] > begin:
                            spans[j][0] += len_diff
                        if spans[j][1] > begin:
                            spans[j][1] += len_diff

            text = " ".join(list(text.replace(" ", "_")))
            candidates = []
            for cand in candidate_str.split(";"):
                candidate = cand
                if cand[0] in ["#", "*", "&"]:
                    candidate = cand[1:]
                candidate = " ".join(list(candidate.replace(" ", "_")))
                candidates.append(candidate)

            span_info = []
            for span in spans:
                span_info.append("CUSTOM " + str(span[0]) + " " + str(span[1]))

            output.write(text + "\t" + ";".join(candidates) + "\t" + target_str + "\t" + ";".join(span_info) + "\n")

        else:
            print("Bad format: ", line)

    output.close()
    input_positive.close()
    input_negative.close()


if __name__ == "__main__":
    main()
