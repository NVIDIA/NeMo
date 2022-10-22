import math
import random
from argparse import ArgumentParser
from collections import defaultdict
from os.path import join
from typing import Dict, Optional, TextIO, Tuple

import numpy as np
from numba import jit

parser = ArgumentParser(
    description="Prepare training examples for Bert: insert custom phrases and best candidates into sample sentences"
)
parser.add_argument("--input_file", required=True, type=str, help="Path to input file with asr hypotheses")
parser.add_argument("--input_vocab", type=str, required=True, help="Path to custom vocabulary")
parser.add_argument("--ngram_mapping", type=str, required=True, help="Path to ngram mapping vocabulary")
parser.add_argument("--output_name", type=str, required=True, help="Output file")

args = parser.parse_args()


def read_custom_vocab():
    phrases = set()
    with open(args.input_vocab, "r", encoding="utf-8") as f:
        for line in f:
            phrases.add(" ".join(list(line.strip().casefold().replace(" ", "_"))))
    return list(phrases)


def get_index(custom_phrases) -> None:
    """Given a restricted vocabulary of replacements,
    loops through custom phrases,
    generates all possible conversions and creates index.
    """

    # load vocab from file
    vocab = defaultdict(dict)
    ban_ngram = set()

    with open(args.ngram_mapping, "r", encoding="utf-8") as f:
        for line in f:
            src, dst, joint_freq, src_freq, dst_freq = line.strip().split("\t")
            assert src != "" and dst != "", "src=" + src + "; dst=" + dst
            dst = dst.replace("<DELETE>", "=")
            if dst.strip() == "":
                continue
            if int(dst_freq) > 10000:
                ban_ngram.add(dst)
            vocab[src][dst] = int(joint_freq) / int(src_freq)

    index_freq = defaultdict(int)
    ngram_to_phrase_and_position = defaultdict(list)

    for custom_phrase in custom_phrases:
        inputs = custom_phrase.split(" ")
        begin = 0
        index_keys = [{} for i in inputs]  # key - letter ngram, index - beginning positions in phrase

        for begin in range(len(inputs)):
            for end in range(begin + 1, min(len(inputs) + 1, begin + 5)):
                inp = " ".join(inputs[begin:end])
                if inp not in vocab:
                    continue
                for rep in vocab[inp]:
                    lp = math.log(vocab[inp][rep])

                    for b in range(max(0, end - 5), end):  # try to grow previous ngrams with new replacement
                        new_ngrams = {}
                        for ngram in index_keys[b]:
                            lp_prev = index_keys[b][ngram]
                            if len(ngram) + len(rep) <= 10 and b + ngram.count(" ") == begin:
                                if lp_prev + lp > -4.0:
                                    new_ngrams[ngram + rep + " "] = lp_prev + lp
                        index_keys[b] = index_keys[b] | new_ngrams  #  join two dictionaries
                    # add current replacement as ngram
                    if lp > -4.0:
                        index_keys[begin][rep + " "] = lp

        for b in range(len(index_keys)):
            for ngram, lp in sorted(index_keys[b].items(), key=lambda item: item[1], reverse=True):
                if ngram in ban_ngram:
                    continue
                real_length = ngram.count(" ")
                ngram = ngram.replace("+", " ").replace("=", " ")
                ngram = " ".join(ngram.split())
                index_freq[ngram] += 1
                if ngram in ban_ngram:
                    continue
                ngram_to_phrase_and_position[ngram].append((custom_phrase, b, real_length, lp))
                if len(ngram_to_phrase_and_position[ngram]) > 100:
                    ban_ngram.add(ngram)
                    del ngram_to_phrase_and_position[ngram]
                    continue

    phrases = []  # id to phrase
    phrase2id = {}  # phrase to id
    ngram2phrases = defaultdict(list)  # ngram to list of phrase ids

    for ngram, freq in sorted(index_freq.items(), key=lambda item: item[1], reverse=True):
        for phrase, b, length, lp in ngram_to_phrase_and_position[ngram]:
            if phrase not in phrase2id:
                phrases.append(phrase)
                phrase2id[phrase] = len(phrases) - 1
            ngram2phrases[ngram].append((phrase2id[phrase], b, length, lp))

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


custom_phrases = read_custom_vocab()
phrases, ngram2phrases = get_index(custom_phrases)

print("len(phrases)=", len(phrases), "; len(ngram2phrases)=", len(ngram2phrases))

with open(args.output_name + ".index", "w", encoding="utf-8") as out_debug:
    for ngram in ngram2phrases:
        for phrase_id, b, size, lp in ngram2phrases[ngram]:
            phr = phrases[phrase_id]
            out_debug.write(ngram + "\t" + phr + "\t" + str(b) + "\t" + str(size) + "\t" + str(lp) + "\n")

dummy_candidates = [
    "a g k t t r k n a p r t f",
    "v w w x y x u r t g p w q",
    "n t r y t q q r u p t l n t",
    "p b r t u r e t f v w x u p z",
    "p p o j j k l n b f q t",
    "j k y u i t d s e w s r e j h i p p",
    "q w r e s f c t d r q g g y",
]
out_debug = open(args.output_name + ".candidates", "w", encoding="utf-8")
out = open(args.output_name, "w", encoding="utf-8")
with open(args.input_file, "r", encoding="utf-8") as f:
    for line in f:
        short_sent, _ = line.strip().split("\t")
        sent = "_".join(short_sent.split())
        letters = list(sent)

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

        candidates = []
        k = 0
        correct_id = 0
        out_debug.write(" ".join(letters) + "\n")
        for idx, coverage in sorted(enumerate(candidate2coverage), key=lambda item: item[1], reverse=True):
            k += 1
            if k > 10:
                break
            if coverage < 0.4:
                candidates.append(random.choice(dummy_candidates))
            else:
                candidates.append(phrases[idx])
                out_debug.write(
                    "\t" + phrases[idx] + "\n" + " ".join(list(map(str, (map(int, phrases2positions[idx]))))) + "\n"
                )

        random.shuffle(candidates)
        if len(candidates) != 10:
            print("WARNING: cannot get 10 candidates", candidates)
            continue
        out.write(" ".join(letters) + "\t" + ";".join(candidates) + "\n")
out.close()
out_debug.close()
