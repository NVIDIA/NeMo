import json
import random
from argparse import ArgumentParser
from collections import defaultdict
from os.path import join
from typing import Dict, Optional, TextIO, Tuple

import numpy as np
from numba import jit
from tqdm.auto import tqdm

## !!!this is temporary hack for my windows machine since is misses some installs 
sys.path.insert(1, "D:\\data\\work\\nemo\\nemo\\collections\\nlp\\data\\spellchecking_asr_customization")
print(sys.path)
from utils import get_all_candidates_coverage, load_index, search_in_index
# from nemo.collections.nlp.data.spellchecking_asr_customization.utils import get_all_candidates_coverage, load_index, search_in_index


parser = ArgumentParser(description="Prepare input for testing search: insert custom phrases into sample sentences")
parser.add_argument("--index_name", required=True, type=str, help='Path to file with index')
parser.add_argument("--output_name", type=str, required=True, help="Output file")
parser.add_argument("--input_name", type=str, required=True, help="Path to simulated input")

args = parser.parse_args()


phrases, ngram2phrases = load_index(args.index_name)

out = open(args.output_name, "w", encoding="utf-8")
out_hard = open(args.output_name + ".hard", "w", encoding="utf-8")
n = 0
correct = 0
with open(args.input_name, "r", encoding="utf-8") as f:
    for line in f:
        n += 1
        if n % 100 == 0:
            print(n)
        sent, reference, position, length = line.strip().split("\t")
        letters = sent.split()
        out.write(line + "\n")

        phrases2positions, position2ngrams = search_in_index(ngram2phrases, phrases, letters)
        candidate2coverage, candidate2position = get_all_candidates_coverage(phrases, phrases2positions)

        k = 0
        found = 0
        for idx, coverage in sorted(enumerate(candidate2coverage), key=lambda item: item[1], reverse=True):
            if k < 20:
                out.write(
                    "\t\t"
                    + phrases[idx]
                    + "\t"
                    + str(coverage)
                    + "\t"
                    + str(candidate2position[idx])
                    + "\t"
                    + reference
                    + "\n"
                )
                if phrases[idx] == reference:
                    correct += 1
                    found = 1
            k += 1
            if k > 20:
                if found:
                    break
                if phrases[idx] == reference:
                    out.write(
                        "\t***"
                        + str(k)
                        + "\t"
                        + phrases[idx]
                        + "\t"
                        + str(coverage)
                        + "\t"
                        + str(candidate2position[idx])
                        + "\t"
                        + reference
                        + "\n"
                    )
                    out_hard.write(line)
                    found = 1
                if k > 200:
                    break
        if not found:
            out.write("\t***NOT FOUND***\t" + reference + "\n")
            out_hard.write(line)


out.write("Correct=" + str(correct) + "\n")
out.write("Total=" + str(n) + "\n")
out.write("Accuracy=" + str(correct / n) + "\n")

out.close()
out_hard.close()
