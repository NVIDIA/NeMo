import sys
import os
from argparse import ArgumentParser
from typing import DefaultDict
import regex as re
import glob
from tqdm import tqdm
from collections import defaultdict
import numpy as np


parser = ArgumentParser(description="TN upsampling")
parser.add_argument("--input_dir", required=True, type=str, help='Path to input directory')
parser.add_argument("--output_path", required=True, type=str, help='Path to input directory')
parser.add_argument("--thresh", default=2000, type=int, help='minimum number per pattern')
parser.add_argument("--pretty", action="store_true", help='Path to input directory')
args = parser.parse_args()


MONEY_PATTERNS = defaultdict(int)
MEASURE_PATTERNS = defaultdict(int)
TIME_PATTERNS = defaultdict(int)
FRACTION_PATTERNS = defaultdict(int)

money_patterns=["([0-9]|\.|,)+"]
measure_patterns=["^-?([0-9]|\.|,|/|\s)+"]
time_patterns=["^[0-9]+:[0-9][0-9]$", "^[0-9]+:[0-9][0-9]\s?[a-zA-Z]+$",  "^[0-9]+\s(p|P|A|a)\.?(m|M)\.?",  "^[0-9]+(p|P|A|a)\.?(m|M)\.?", "^[0-9]:[0-9][0-9]\s(p|P|A|a)\.?(m|M)\.?", "^[0-9][0-9]:[0-9][0-9]\s(p|P|A|a)\.?(m|M)\.?", "^[0-9]:[0-9][0-9](p|P|A|a)\.?(m|M)\.?", "^[0-9][0-9]:[0-9][0-9](p|P|A|a)\.?(m|M)\.?", "^[0-9]+.[0-9][0-9]\s?(p|P|A|a)\.?(m|M)\.?", "^[0-9]+:[0-9]+:[0-9]+", "^[0-9]+:[0-9]+.[0-9]+", "^[0-9]+.[0-9]+$", "^[0-9]+.[0-9]+\s?[a-zA-Z]+$"]
fraction_patterns=["^-?[0-9]+\s?\/\s?[0-9]{3}$", "^-?[0-9]{3}\s?\/\s?[0-9]+$", "^[0-9]+\s[0-9]+\/[0-9]+$", "^[0-9]+\s[0-9]+\/[0-9]+$", "^[0-9]+\s[0-9]+\s\/\s[0-9]+$", "^-?[0-9]+\s\/\s[0-9]+$", "^-?[0-9]+\/[0-9]+$"]

classes_to_upsample = defaultdict(int)

def include_sentence(sentence_patterns):
    include = False
    for k, v in sentence_patterns["MONEY"].items():
        if v > 0 and k in MONEY_PATTERNS and MONEY_PATTERNS[k] < args.thresh:
            include = True
    for k, v in sentence_patterns["MEASURE"].items():
        if v > 0 and k in MEASURE_PATTERNS and MEASURE_PATTERNS[k] < args.thresh:
            include = True
    for k, v in sentence_patterns["TIME"].items():
        if v > 0 and k in TIME_PATTERNS and TIME_PATTERNS[k] < args.thresh:
            include = True
    for k, v in sentence_patterns["FRACTION"].items():
        if v > 0 and k in FRACTION_PATTERNS and FRACTION_PATTERNS[k] < args.thresh:
            include = True

    if include:
        for k, v in sentence_patterns["MONEY"].items():
            if v > 0 and k in MONEY_PATTERNS:
                MONEY_PATTERNS[k] += v
                if MONEY_PATTERNS[k]-v < args.thresh and MONEY_PATTERNS[k] >= args.thresh:
                    classes_to_upsample["MONEY"] -= 1
                    if classes_to_upsample["MONEY"] <= 0:
                        classes_to_upsample.pop("MONEY")
        for k, v in sentence_patterns["MEASURE"].items():
            if v > 0 and k in MEASURE_PATTERNS:
                MEASURE_PATTERNS[k] += v
                if MEASURE_PATTERNS[k]-v < args.thresh and MEASURE_PATTERNS[k] >= args.thresh:
                    classes_to_upsample["MEASURE"] -= 1
                    if classes_to_upsample["MEASURE"] <= 0:
                        classes_to_upsample.pop("MEASURE")
        for k, v in sentence_patterns["TIME"].items():
            if v > 0 and k in TIME_PATTERNS:
                TIME_PATTERNS[k] += v
                if TIME_PATTERNS[k]-v < args.thresh and TIME_PATTERNS[k] >= args.thresh:
                    classes_to_upsample["TIME"] -= 1
                    if classes_to_upsample["TIME"] <= 0:
                        classes_to_upsample.pop("TIME")
        for k, v in sentence_patterns["FRACTION"].items():
            if v > 0 and k in FRACTION_PATTERNS:
                FRACTION_PATTERNS[k] += v
                if FRACTION_PATTERNS[k]-v < args.thresh and FRACTION_PATTERNS[k] >= args.thresh:
                    classes_to_upsample["FRACTION"] -= 1
                    if classes_to_upsample["FRACTION"] <= 0:
                        classes_to_upsample.pop("FRACTION")
    return include

def read_data_file(fp, sample=False):
    """ Reading the raw data from a file of NeMo format
    For more info about the data format, refer to the
    `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`.

    Args:
        fp: file paths
    Returns:
        insts: List of sentences parsed as list of words
    """

    insts, w_words, s_words, classes = [], [], [], []
    # Read input file
    with open(fp, 'r', encoding='utf-8') as f:
        sentence_patterns = {"FRACTION": defaultdict(int), "MEASURE": defaultdict(int),"TIME": defaultdict(int),"MONEY": defaultdict(int),}
        for line in f:
            es = [e.strip() for e in line.strip().split('\t')]
            if es[0] == '<eos>':
                if not sample:
                    inst = (classes, w_words, s_words)
                    insts.append(inst)
                else:
                    ok = include_sentence(sentence_patterns) 
                    if ok:
                        inst = (classes, w_words, s_words)
                        insts.append(inst)
                    
                # Reset
                w_words, s_words, classes = [], [], []
                sentence_patterns = {"FRACTION": defaultdict(int), "MEASURE": defaultdict(int),"TIME": defaultdict(int),"MONEY": defaultdict(int),}
                
            else:
                classes.append(es[0])
                w_words.append(es[1])
                s_words.append(es[2])
                if not sample:
                    register_patterns(es[0], es[1], es[2], pretty=args.pretty)
                else:
                    if es[0] in classes_to_upsample:
                        patterns = lookup_patterns(es[0], es[1], es[2])
                        update_patterns(sentence_patterns[es[0]], patterns)


        if not sample:
            inst = (classes, w_words, s_words)
            insts.append(inst)
    return insts




def count_pattern(inst, cls, pattern):
    c = 0
    for sent in inst:
        for x_cls, x_written, x_spoken in sent:
            if x_cls == cls and x_written.search(pattern) is not None:
                c += 1
    return c

def update_patterns(patterns, new_patterns):
    for k, v in new_patterns.items():
        patterns[k] += v

def register_patterns(cls, written, spoken, pretty=False):
    if cls == "MONEY":
        new_dict = create_pattern(money_patterns, written, pretty=pretty)
        update_patterns(MONEY_PATTERNS, new_dict)
    if cls == "MEASURE":
        new_dict = create_pattern(measure_patterns, written, pretty=pretty)
        update_patterns(MEASURE_PATTERNS, new_dict)
    if cls == "TIME":
        new_dict = create_pattern(time_patterns, written, pretty=pretty)
        update_patterns(TIME_PATTERNS, new_dict)
    if cls == "FRACTION":
        new_dict = create_pattern(fraction_patterns, written, pretty=pretty)
        update_patterns(FRACTION_PATTERNS, new_dict)


def lookup_patterns(cls, written, spoken):
    if cls == "MONEY":
        new_dict = create_pattern(MONEY_PATTERNS.keys(), written)
    if cls == "MEASURE":
        new_dict = create_pattern(MEASURE_PATTERNS.keys(), written)
    if cls == "TIME":
        new_dict = create_pattern(TIME_PATTERNS.keys(), written)
    if cls == "FRACTION":
        new_dict = create_pattern(FRACTION_PATTERNS.keys(), written)
    return new_dict
    

def create_pattern(patterns, written, pretty=False):
    res = defaultdict(int)
    for pattern in patterns:
        if re.search(pattern, written) is None:
            continue
        if not pretty:
            res[re.sub(pattern, pattern, written)] += 1
        else:
            res[re.sub(pattern, "@", written)] += 1
    return res

def print_stats():
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
    print("READING", input_files[0])
    inst_first_file = read_data_file(input_files[0])

    measure_keys = list(MEASURE_PATTERNS.keys())
    for k in measure_keys:
        if re.search("\s?st$", k) is not None:
            MEASURE_PATTERNS.pop(k)
    classes_to_upsample["FRACTION"] = sum(np.asarray(list(FRACTION_PATTERNS.values())) < args.thresh)
    classes_to_upsample["MEASURE"] = sum(np.asarray(list(MEASURE_PATTERNS.values())) < args.thresh)
    classes_to_upsample["TIME"] = sum(np.asarray(list(TIME_PATTERNS.values())) < args.thresh)
    classes_to_upsample["MONEY"] = sum(np.asarray(list(MONEY_PATTERNS.values())) < args.thresh)

    print_stats()

    for fp in input_files[1:]:
        print("READING", fp)
        instances = read_data_file(fp, sample=True)
        inst_first_file.extend(instances)
        
        print_stats()

    output_f = open(args.output_path, 'w+', encoding='utf-8')
    for ix, inst in enumerate(inst_first_file):
        cur_classes, cur_tokens, cur_outputs = inst
        for c, t, o in zip(cur_classes, cur_tokens, cur_outputs):
            output_f.write(f'{c}\t{t}\t{o}\n')

        output_f.write(f'<eos>\t<eos>\n')







if __name__=="__main__":
    main()

