import json
import random
from argparse import ArgumentParser
from collections import defaultdict
from os.path import join
from typing import Dict, Optional, TextIO, Tuple

import numpy as np
from numba import jit
from tqdm.auto import tqdm

parser = ArgumentParser(description="Prepare input for testing search: insert custom phrases into sample sentences")
parser.add_argument("--index_name", required=True, type=str, help='Path to file with index')
parser.add_argument("--output_name", type=str, required=True, help="Output file")
parser.add_argument("--input_name", type=str, required=True, help="Path to simulated input")

args = parser.parse_args()


def read_index():
    phrases = []  # id to phrase
    phrase2id = {}  # phrase to id
    ngram2phrases = defaultdict(list)  # ngram to list of phrase ids
    with open(args.index_name, "r", encoding="utf-8") as f:
        for line in f:
            ngram, phrase, begin, length, lp = line.strip().split("\t")
            if phrase not in phrase2id:
                phrases.append(phrase)
                phrase2id[phrase] = len(phrases) - 1
            ngram2phrases[ngram].append((phrase2id[phrase], int(begin), int(length), float(lp)))
    return phrases, ngram2phrases


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def get_all_candidates_coverage(phrases, phrases2positions):
    candidate2coverage = [0.0] * len(phrases)
    candidate2position = [-1] * len(phrases)

    for i in range(len(phrases)):
        phrase_length = phrases[i].count(" ") + 1
        all_coverage = np.sum(phrases2positions[i]) / phrase_length
        if all_coverage < 0.4:
            continue
        moving_sum = np.sum(phrases2positions[i, 0:phrase_length])
        max_sum = moving_sum
        best_pos = 0
        for pos in range(1, phrases2positions.shape[1] - phrase_length):
            moving_sum -= phrases2positions[i, pos - 1]
            moving_sum += phrases2positions[i, pos + phrase_length - 1]
            if moving_sum > max_sum:
                max_sum = moving_sum
                best_pos = pos

        coverage = max_sum / (phrase_length + 2)  # smoothing
        candidate2coverage[i] = coverage
        candidate2position[i] = best_pos
    return candidate2coverage, candidate2position


phrases, ngram2phrases = read_index()

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

        phrases2positions = np.zeros((len(phrases), len(letters)), dtype=float)
        position2ngrams = [{}] * len(letters)  # positions mapped to dicts of ngrams starting from that position

        begin = 0
        for begin in range(len(letters)):
            for end in range(begin + 1, min(len(letters) + 1, begin + 7)):
                ngram = " ".join(letters[begin:end])
                if ngram not in ngram2phrases:
                    continue
                for phrase_id, b, size, lp in ngram2phrases[ngram]:
                    phrases2positions[phrase_id, begin:end] = 1.0

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
