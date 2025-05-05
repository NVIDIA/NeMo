# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script can be used to create a more class balanced file from a set of the data files of the English Google Text Normalization dataset
for better training performance. Currently this script upsamples the class types "MONEY", "MEASURE", "TIME", "FRACTION" since these are underrepresented in the Google Text Normalization dataset, but still diverse in its representations.
Of all the input files in `input_dir` this script takes the first file and computes the class patterns that occurs in it.
For those that are underrepresented, quantitatively defined as lower than `min_number`, the other files are scanned for sentences that have the missing patterns. 
Those sentences are appended to the first file and outputted. 

USAGE Example:
1. Download the Google TN dataset from https://www.kaggle.com/google-nlu/text-normalization
2. Unzip the English subset (e.g., by running `tar zxvf  en_with_types.tgz`). Then there will a folder named `en_with_types`.
3. Run the data_split.py, data_preprocessing.py scripts to obtain cleaned data files
4. Run this script on the training data portion
# python upsample.py       \
        --input_dir=train_processed/           \
        --output_file=train_upsampled.tsv/  \
        --min_number=2000 

In this example, the final file will be train_upsampled.tsv.
"""


import glob
from argparse import ArgumentParser
from collections import defaultdict
from typing import List

import numpy as np
import regex as re

parser = ArgumentParser(description="English Text Normalization upsampling")
parser.add_argument("--input_dir", required=True, type=str, help='Path to input directory with preprocessed data')
parser.add_argument("--output_file", required=True, type=str, help='Path to output file')
parser.add_argument("--min_number", default=2000, type=int, help='minimum number per pattern')
parser.add_argument("--pretty", action="store_true", help='Pretty print')
args = parser.parse_args()

# global pattern tables
MONEY_PATTERNS = defaultdict(int)
MEASURE_PATTERNS = defaultdict(int)
TIME_PATTERNS = defaultdict(int)
FRACTION_PATTERNS = defaultdict(int)

# global templates/stencils for creating patterns
money_templates = ["([0-9]|\.|,)+"]
measure_templates = ["^-?([0-9]|\.|,|/|\s)+"]
time_templates = [
    "^[0-9]+:[0-9][0-9]$",
    "^[0-9]+:[0-9][0-9]\s?[a-zA-Z]+$",
    "^[0-9]+\s(p|P|A|a)\.?(m|M)\.?",
    "^[0-9]+(p|P|A|a)\.?(m|M)\.?",
    "^[0-9]:[0-9][0-9]\s(p|P|A|a)\.?(m|M)\.?",
    "^[0-9][0-9]:[0-9][0-9]\s(p|P|A|a)\.?(m|M)\.?",
    "^[0-9]:[0-9][0-9](p|P|A|a)\.?(m|M)\.?",
    "^[0-9][0-9]:[0-9][0-9](p|P|A|a)\.?(m|M)\.?",
    "^[0-9]+.[0-9][0-9]\s?(p|P|A|a)\.?(m|M)\.?",
    "^[0-9]+:[0-9]+:[0-9]+",
    "^[0-9]+:[0-9]+.[0-9]+",
    "^[0-9]+.[0-9]+$",
    "^[0-9]+.[0-9]+\s?[a-zA-Z]+$",
]
fraction_templates = [
    "^-?[0-9]+\s?\/\s?[0-9]{3}$",
    "^-?[0-9]{3}\s?\/\s?[0-9]+$",
    "^[0-9]+\s[0-9]+\/[0-9]+$",
    "^[0-9]+\s[0-9]+\/[0-9]+$",
    "^[0-9]+\s[0-9]+\s\/\s[0-9]+$",
    "^-?[0-9]+\s\/\s[0-9]+$",
    "^-?[0-9]+\/[0-9]+$",
]

# classes that still need to be upsampled, and required number of instances needed
classes_to_upsample = defaultdict(int)


def include_sentence(sentence_patterns) -> bool:
    """
    Determines whether to use a sentence for upsampling whose patterns are provided as input. This will check the global pattern tables 
    if this sentence includes any patterns that are still needed.

    Args:
        sentence_patterns: dictionary of patterns for a sentence grouped by class
    Returns:
        include: whether or not to use the sentence or for upsampling
    """
    include = False
    for k, v in sentence_patterns["MONEY"].items():
        if v > 0 and k in MONEY_PATTERNS and MONEY_PATTERNS[k] < args.min_number:
            include = True
    for k, v in sentence_patterns["MEASURE"].items():
        if v > 0 and k in MEASURE_PATTERNS and MEASURE_PATTERNS[k] < args.min_number:
            include = True
    for k, v in sentence_patterns["TIME"].items():
        if v > 0 and k in TIME_PATTERNS and TIME_PATTERNS[k] < args.min_number:
            include = True
    for k, v in sentence_patterns["FRACTION"].items():
        if v > 0 and k in FRACTION_PATTERNS and FRACTION_PATTERNS[k] < args.min_number:
            include = True

    if include:
        for k, v in sentence_patterns["MONEY"].items():
            if v > 0 and k in MONEY_PATTERNS:
                MONEY_PATTERNS[k] += v
                if MONEY_PATTERNS[k] - v < args.min_number and MONEY_PATTERNS[k] >= args.min_number:
                    classes_to_upsample["MONEY"] -= 1
                    if classes_to_upsample["MONEY"] <= 0:
                        classes_to_upsample.pop("MONEY")
        for k, v in sentence_patterns["MEASURE"].items():
            if v > 0 and k in MEASURE_PATTERNS:
                MEASURE_PATTERNS[k] += v
                if MEASURE_PATTERNS[k] - v < args.min_number and MEASURE_PATTERNS[k] >= args.min_number:
                    classes_to_upsample["MEASURE"] -= 1
                    if classes_to_upsample["MEASURE"] <= 0:
                        classes_to_upsample.pop("MEASURE")
        for k, v in sentence_patterns["TIME"].items():
            if v > 0 and k in TIME_PATTERNS:
                TIME_PATTERNS[k] += v
                if TIME_PATTERNS[k] - v < args.min_number and TIME_PATTERNS[k] >= args.min_number:
                    classes_to_upsample["TIME"] -= 1
                    if classes_to_upsample["TIME"] <= 0:
                        classes_to_upsample.pop("TIME")
        for k, v in sentence_patterns["FRACTION"].items():
            if v > 0 and k in FRACTION_PATTERNS:
                FRACTION_PATTERNS[k] += v
                if FRACTION_PATTERNS[k] - v < args.min_number and FRACTION_PATTERNS[k] >= args.min_number:
                    classes_to_upsample["FRACTION"] -= 1
                    if classes_to_upsample["FRACTION"] <= 0:
                        classes_to_upsample.pop("FRACTION")
    return include


def read_data_file(fp: str, upsample_file: bool = False):
    """ Reading the raw data from a file of NeMo format
    For more info about the data format, refer to the
    `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`.

    Args:
        fp: file paths
        upsample_file: whether or not this input file should be used in full or only for upsampling, i.e. only as a subset
    Returns:
        insts: List of sentences parsed as list of words
    """

    insts, w_words, s_words, classes = [], [], [], []
    with open(fp, 'r', encoding='utf-8') as f:
        sentence_patterns = {
            "FRACTION": defaultdict(int),
            "MEASURE": defaultdict(int),
            "TIME": defaultdict(int),
            "MONEY": defaultdict(int),
        }
        for line in f:
            es = [e.strip() for e in line.strip().split('\t')]
            if es[0] == '<eos>':
                if not upsample_file:
                    inst = (classes, w_words, s_words)
                    insts.append(inst)
                else:
                    ok = include_sentence(sentence_patterns)
                    if ok:
                        inst = (classes, w_words, s_words)
                        insts.append(inst)
                # Reset
                w_words, s_words, classes = [], [], []
                sentence_patterns = {
                    "FRACTION": defaultdict(int),
                    "MEASURE": defaultdict(int),
                    "TIME": defaultdict(int),
                    "MONEY": defaultdict(int),
                }

            else:
                classes.append(es[0])
                w_words.append(es[1])
                s_words.append(es[2])
                if not upsample_file:
                    register_patterns(cls=es[0], input_str=es[1], pretty=args.pretty)
                else:
                    if es[0] in classes_to_upsample:
                        patterns = lookup_patterns(cls=es[0], input_str=es[1])
                        update_patterns(sentence_patterns[es[0]], patterns)
        if not upsample_file:
            inst = (classes, w_words, s_words)
            insts.append(inst)
    return insts


def update_patterns(patterns: dict, new_patterns: dict):
    """
    updates a given pattern table with counts from another table by adding them to the given table.

    Args:
        patterns: main table
        new_patterns: new table to update the main table with 
    """
    for k, v in new_patterns.items():
        patterns[k] += v


def register_patterns(cls: str, input_str: str, pretty: bool = False):
    """
    Saves all patterns created from input string from global templates/stencils to global pattern table

    Args:
        cls: class type of input_str
        input_str: input string
        pretty: used to pretty print patterns
    """
    if cls == "MONEY":
        new_dict = create_pattern(money_templates, input_str, pretty=pretty)
        update_patterns(MONEY_PATTERNS, new_dict)
    if cls == "MEASURE":
        new_dict = create_pattern(measure_templates, input_str, pretty=pretty)
        update_patterns(MEASURE_PATTERNS, new_dict)
    if cls == "TIME":
        new_dict = create_pattern(time_templates, input_str, pretty=pretty)
        update_patterns(TIME_PATTERNS, new_dict)
    if cls == "FRACTION":
        new_dict = create_pattern(fraction_templates, input_str, pretty=pretty)
        update_patterns(FRACTION_PATTERNS, new_dict)


def lookup_patterns(cls: str, input_str: str) -> dict:
    """
    Look up all patterns that match an input string from global pattern table

    Args:
        cls: class type of input_str
        input_str: input string
    """
    if cls == "MONEY":
        new_dict = create_pattern(MONEY_PATTERNS.keys(), input_str)
    if cls == "MEASURE":
        new_dict = create_pattern(MEASURE_PATTERNS.keys(), input_str)
    if cls == "TIME":
        new_dict = create_pattern(TIME_PATTERNS.keys(), input_str)
    if cls == "FRACTION":
        new_dict = create_pattern(FRACTION_PATTERNS.keys(), input_str)
    return new_dict


def create_pattern(templates: List[str], input_str: str, pretty: bool = False):
    """
    create all patterns based on list of input templates using the input string. 

    Args:
        templates: list of templates/stencils
        input_str: string to apply templates on to create patterns
        pretty: used to pretty print patterns
    """
    res = defaultdict(int)
    for template in templates:
        if re.search(template, input_str) is None:
            continue
        if not pretty:
            res[re.sub(template, template, input_str)] += 1
        else:
            res[re.sub(template, "@", input_str)] += 1
    return res


def print_stats():
    """
    print statistics on class patterns to be upsampled
    """
    print("MONEY")
    for k, v in MONEY_PATTERNS.items():
        print(f"\t{k}\t{v}")
    print("no. patterns to upsample", classes_to_upsample["MONEY"])
    print("MEASURE")
    for k, v in MEASURE_PATTERNS.items():
        print(f"\t{k}\t{v}")
    print("no. patterns to upsample", classes_to_upsample["MEASURE"])
    print("TIME")
    for k, v in TIME_PATTERNS.items():
        print(f"\t{k}\t{v}")
    print("no. patterns to upsample", classes_to_upsample["TIME"])
    print("FRACTION")
    for k, v in FRACTION_PATTERNS.items():
        print(f"\t{k}\t{v}")
    print("no. patterns to upsample", classes_to_upsample["FRACTION"])


def main():
    input_files = sorted(glob.glob(f"{args.input_dir}/output-*"))
    print("Taking in full: ", input_files[0])
    inst_first_file = read_data_file(input_files[0])

    measure_keys = list(MEASURE_PATTERNS.keys())
    for k in measure_keys:
        if re.search("\s?st$", k) is not None or re.search("\s?Da$", k) is not None:
            MEASURE_PATTERNS.pop(k)

    money_keys = list(MONEY_PATTERNS.keys())
    for k in money_keys:
        if re.search("(DM|SHP|BMD|SCR|SHP|ARS|BWP|SBD)$", k) is not None:
            MONEY_PATTERNS.pop(k)

    classes_to_upsample["FRACTION"] = sum(np.asarray(list(FRACTION_PATTERNS.values())) < args.min_number)
    classes_to_upsample["MEASURE"] = sum(np.asarray(list(MEASURE_PATTERNS.values())) < args.min_number)
    classes_to_upsample["TIME"] = sum(np.asarray(list(TIME_PATTERNS.values())) < args.min_number)
    classes_to_upsample["MONEY"] = sum(np.asarray(list(MONEY_PATTERNS.values())) < args.min_number)

    print_stats()
    for fp in input_files[1:]:
        print("Upsamling: ", fp)
        instances = read_data_file(fp, upsample_file=True)
        inst_first_file.extend(instances)
        print_stats()

    output_f = open(args.output_file, 'w+', encoding='utf-8')
    for ix, inst in enumerate(inst_first_file):
        cur_classes, cur_tokens, cur_outputs = inst
        for c, t, o in zip(cur_classes, cur_tokens, cur_outputs):
            output_f.write(f'{c}\t{t}\t{o}\n')
        output_f.write(f'<eos>\t<eos>\n')


if __name__ == "__main__":
    main()
