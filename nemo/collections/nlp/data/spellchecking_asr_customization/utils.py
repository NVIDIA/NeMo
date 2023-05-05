# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


import json
import math
import re
from collections import defaultdict, namedtuple
from heapq import heappush, heapreplace
from typing import Dict, List, Set, Tuple

import numpy as np
from numba import jit

"""Utility functions for Spellchecking ASR Customization."""


def replace_diacritics(text):
    text = re.sub(r"[éèëēêęěė]", "e", text)  # latin
    text = re.sub(r"[ё]", "е", text)  # cyrillic
    text = re.sub(r"[ãâāáäăâàąåạảǎ]", "a", text)
    text = re.sub(r"[úūüùưûů]", "u", text)
    text = re.sub(r"[ôōóöõòőø]", "o", text)
    text = re.sub(r"[ćçč]", "c", text)
    text = re.sub(r"[ïīíîıì]", "i", text)
    text = re.sub(r"[ñńňņ]", "n", text)
    text = re.sub(r"[țťţ]", "t", text)
    text = re.sub(r"[łľļ]", "l", text)
    text = re.sub(r"[żžź]", "z", text)
    text = re.sub(r"[ğ]", "g", text)
    text = re.sub(r"[ďđ]", "d", text)
    text = re.sub(r"[ķ]", "k", text)
    text = re.sub(r"[ř]", "r", text)
    text = re.sub(r"[ý]", "y", text)
    text = re.sub(r"[æ]", "ae", text)
    text = re.sub(r"[œ]", "oe", text)
    text = re.sub(r"[șşšś]", "s", text)
    return text


def load_ngram_mappings(input_name: str, max_dst_freq: int = 1000000000) -> Tuple[defaultdict, Set]:
    """Loads vocab from file
    Input format:
        u t o	u+i t o	49	8145	114
        u t o	<DELETE> t e	63	8145	16970
        u t o	o+_ t o	42	8145	1807
    """
    vocab = defaultdict(dict)
    ban_ngram = set()

    with open(input_name, "r", encoding="utf-8") as f:
        for line in f:
            src, dst, joint_freq, src_freq, dst_freq = line.strip().split("\t")
            assert src != "" and dst != "", "src=" + src + "; dst=" + dst
            dst = dst.replace("<DELETE>", "=")
            if dst.replace("=", "").strip() == "":  # skip if resulting ngram doesn't contain any real character
                continue
            if int(dst_freq) > max_dst_freq:
                ban_ngram.add(dst + " ")  # space at the end is required within get_index function
            vocab[src][dst] = int(joint_freq) / int(src_freq)
    return vocab, ban_ngram


def load_ngram_mappings_for_dp(input_name: str) -> Tuple[defaultdict, defaultdict, defaultdict, int]:
    """Loads vocab from file
    Input format:
        u t o	u+i t o	49	8145	114
        u t o	<DELETE> t e	63	8145	16970
        u t o	o+_ t o	42	8145	1807
    """
    joint_vocab = defaultdict(int)
    src_vocab = defaultdict(int)
    dst_vocab = defaultdict(int)
    max_len = 0
    with open(input_name, "r", encoding="utf-8") as f:
        for line in f:
            src, dst, joint_freq, _, _ = line.strip().split("\t")
            assert src != "" and dst != "", "src=" + src + "; dst=" + dst
            dst = dst.replace("<DELETE>", " ").replace("+", " ")
            dst = " ".join(dst.split())
            if dst == "":  # skip if resulting ngram doesn't contain any real character
                continue
            max_len = max(max_len, src.count(" ") + 1, dst.count(" ") + 1)
            joint_vocab[(src, dst)] += int(joint_freq)
            src_vocab[src] += int(joint_freq)
            dst_vocab[dst] += int(joint_freq)
    return joint_vocab, src_vocab, dst_vocab, max_len


def get_alignment_by_dp(
    hyp_phrase: str,
    ref_phrase: str,
    joint_vocab: defaultdict,
    src_vocab: defaultdict,
    dst_vocab: defaultdict,
    max_len: int,
) -> List[Tuple[str, str, float, float, int, int, int]]:
    hyp_letters = ["*"] + hyp_phrase.split()
    ref_letters = ["*"] + ref_phrase.split()
    DpInfo = namedtuple(
        "DpInfo", ["hyp_pos", "ref_pos", "best_hyp_ngram_len", "best_ref_ngram_len", "score", "sum_score"]
    )
    history = defaultdict(DpInfo)
    history[(0, 0)] = DpInfo(
        hyp_pos=0, ref_pos=0, best_hyp_ngram_len=1, best_ref_ngram_len=1, score=0.0, sum_score=0.0
    )
    for hyp_pos in range(len(hyp_letters)):
        for ref_pos in range(len(ref_letters)):
            if hyp_pos == 0 and ref_pos == 0:  # cell (0, 0) is already defined
                continue
            # consider cell (hyp_pos, ref_pos) and find best path to get there
            best_hyp_ngram_len = 0
            best_ref_ngram_len = 0
            best_ngram_score = float("-inf")
            best_sum_score = float("-inf")
            # loop over paths ending on non-empty ngram mapping
            for hyp_ngram_len in range(1, 1 + min(max_len, hyp_pos + 1)):
                hyp_ngram = " ".join(hyp_letters[(hyp_pos - hyp_ngram_len + 1) : (hyp_pos + 1)])
                for ref_ngram_len in range(1, 1 + min(max_len, ref_pos + 1)):
                    ref_ngram = " ".join(ref_letters[(ref_pos - ref_ngram_len + 1) : (ref_pos + 1)])
                    if (hyp_ngram, ref_ngram) not in joint_vocab:
                        continue
                    joint_freq = joint_vocab[(hyp_ngram, ref_ngram)]
                    src_freq = src_vocab.get(hyp_ngram, 1)
                    ngram_score = math.log(joint_freq / src_freq)
                    previous_score = 0.0
                    previous_cell = (hyp_pos - hyp_ngram_len, ref_pos - ref_ngram_len)
                    if previous_cell not in history:
                        print("cell ", previous_cell, "does not exist")
                        continue
                    previous_score = history[previous_cell].sum_score
                    sum_score = ngram_score + previous_score
                    if sum_score > best_sum_score:
                        best_sum_score = sum_score
                        best_ngram_score = ngram_score
                        best_hyp_ngram_len = hyp_ngram_len
                        best_ref_ngram_len = ref_ngram_len
            # loop over two variants with deletion of one character
            deletion_score = -6.0
            insertion_score = -6.0
            if hyp_pos > 0:
                previous_cell = (hyp_pos - 1, ref_pos)
                previous_score = history[previous_cell].sum_score
                sum_score = deletion_score + previous_score
                if sum_score > best_sum_score:
                    best_sum_score = sum_score
                    best_ngram_score = deletion_score
                    best_hyp_ngram_len = 1
                    best_ref_ngram_len = 0

            if ref_pos > 0:
                previous_cell = (hyp_pos, ref_pos - 1)
                previous_score = history[previous_cell].sum_score
                sum_score = insertion_score + previous_score
                if sum_score > best_sum_score:
                    best_sum_score = sum_score
                    best_ngram_score = insertion_score
                    best_hyp_ngram_len = 0
                    best_ref_ngram_len = 1

            if best_hyp_ngram_len == 0 and best_ref_ngram_len == 0:
                raise (ValueError, "best_hyp_ngram_len = 0 and best_ref_ngram_len = 0")

            # save cell to history
            history[(hyp_pos, ref_pos)] = DpInfo(
                hyp_pos=hyp_pos,
                ref_pos=ref_pos,
                best_hyp_ngram_len=best_hyp_ngram_len,
                best_ref_ngram_len=best_ref_ngram_len,
                score=best_ngram_score,
                sum_score=best_sum_score,
            )
    # now trace back on best path starting from last positions
    path = []
    hyp_pos = len(hyp_letters) - 1
    ref_pos = len(ref_letters) - 1
    cell_info = history[(hyp_pos, ref_pos)]
    path.append(cell_info)
    while hyp_pos > 0 or ref_pos > 0:
        hyp_pos -= cell_info.best_hyp_ngram_len
        ref_pos -= cell_info.best_ref_ngram_len
        cell_info = history[(hyp_pos, ref_pos)]
        path.append(cell_info)

    result = []
    for info in reversed(path):
        hyp_ngram = " ".join(hyp_letters[(info.hyp_pos - info.best_hyp_ngram_len + 1) : (info.hyp_pos + 1)])
        ref_ngram = " ".join(ref_letters[(info.ref_pos - info.best_ref_ngram_len + 1) : (info.ref_pos + 1)])
        joint_freq = joint_vocab.get((hyp_ngram, ref_ngram), 0)
        src_freq = src_vocab.get(hyp_ngram, 0)
        dst_freq = dst_vocab.get(ref_ngram, 0)
        result.append((hyp_ngram, ref_ngram, info.score, info.sum_score, joint_freq, src_freq, dst_freq))
    return result


def get_index(
    custom_phrases: List[str],
    vocab: defaultdict,
    ban_ngram_global: Set[str],
    min_log_prob: float = -4.0,
    max_phrases_per_ngram: int = 100,
) -> Tuple[List[str], Dict[str, List[Tuple[int, int, int, float]]]]:
    """Given a restricted vocabulary of replacements,
    loops through custom phrases,
    generates all possible conversions and creates index.
    """

    ban_ngram_local = set()  # these ngrams are banned only for given custom_phrases
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
                                if lp_prev + lp > min_log_prob:
                                    new_ngrams[ngram + rep + " "] = lp_prev + lp
                        index_keys[b].update(new_ngrams)  #  join two dictionaries
                    # add current replacement as ngram
                    if lp > min_log_prob:
                        index_keys[begin][rep + " "] = lp

        for b in range(len(index_keys)):
            for ngram, lp in sorted(index_keys[b].items(), key=lambda item: item[1], reverse=True):
                if ngram in ban_ngram_global:  # here ngram ends with a space
                    continue
                real_length = ngram.count(" ")
                ngram = ngram.replace("+", " ").replace("=", " ")
                ngram = " ".join(ngram.split())  # here ngram doesn't end with a space anymore
                if ngram + " " in ban_ngram_global:  # this can happen after deletion of + and =
                    continue
                if ngram in ban_ngram_local:
                    continue
                ngram_to_phrase_and_position[ngram].append((custom_phrase, b, real_length, lp))
                if len(ngram_to_phrase_and_position[ngram]) > max_phrases_per_ngram:
                    ban_ngram_local.add(ngram)
                    del ngram_to_phrase_and_position[ngram]
                    continue

    phrases = []  # id to phrase
    phrase2id = {}  # phrase to id
    ngram2phrases = defaultdict(list)  # ngram to list of tuples (phrase_id, begin, length, logprob)

    for ngram in ngram_to_phrase_and_position:
        for phrase, b, length, lp in ngram_to_phrase_and_position[ngram]:
            if phrase not in phrase2id:
                phrases.append(phrase)
                phrase2id[phrase] = len(phrases) - 1
            ngram2phrases[ngram].append((phrase2id[phrase], b, length, lp))

    return phrases, ngram2phrases


def load_index(input_name: str) -> Tuple[List[str], Dict[str, List[Tuple[int, int, int, float]]]]:
    phrases = []  # id to phrase
    phrase2id = {}  # phrase to id
    ngram2phrases = defaultdict(list)  # ngram to list of phrase ids
    with open(input_name, "r", encoding="utf-8") as f:
        for line in f:
            ngram, phrase, b, size, lp = line.split("\t")
            b = int(b)
            size = int(size)
            lp = float(lp)
            if phrase not in phrase2id:
                phrases.append(phrase)
                phrase2id[phrase] = len(phrases) - 1
            ngram2phrases[ngram].append((phrase2id[phrase], b, size, lp))
    return phrases, ngram2phrases


def search_in_index(
    ngram2phrases: Dict[str, List[Tuple[int, int, int, float]]], phrases: List[str], letters: List[str]
):
    if " " in letters:
        raise ValueError("letters should not contain space: " + str(letters))

    phrases2positions = np.zeros((len(phrases), len(letters)), dtype=float)
    # positions mapped to sets of ngrams starting from that position
    position2ngrams = [set() for _ in range(len(letters))]

    begin = 0
    for begin in range(len(letters)):
        for end in range(begin + 1, min(len(letters) + 1, begin + 7)):
            ngram = " ".join(letters[begin:end])
            if ngram not in ngram2phrases:
                continue
            for phrase_id, b, size, lp in ngram2phrases[ngram]:
                phrases2positions[phrase_id, begin:end] = 1.0
            position2ngrams[begin].add(ngram)
    return phrases2positions, position2ngrams


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
        for pos in range(1, phrases2positions.shape[1] - phrase_length + 1):
            moving_sum -= phrases2positions[i, pos - 1]
            moving_sum += phrases2positions[i, pos + phrase_length - 1]
            if moving_sum > max_sum:
                max_sum = moving_sum
                best_pos = pos

        coverage = max_sum / (phrase_length + 2)  # smoothing
        candidate2coverage[i] = coverage
        candidate2position[i] = best_pos
    return candidate2coverage, candidate2position
