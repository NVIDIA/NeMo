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
import numpy as np
import random
import re
from collections import defaultdict, namedtuple
from heapq import heappush, heapreplace
from numba import jit
from typing import Dict, List, Set, Tuple
import pdb

"""Utility functions for Spellchecking ASR Customization."""


SPACE_REGEX = re.compile(r"[\u2000-\u200F]", re.UNICODE)
APOSTROPHES_REGEX = re.compile(r"[’'‘`ʽ']")
# ATTENTION: do not delete hyphen and apostrophe
CHARS_TO_IGNORE_REGEX = re.compile(r"[\.\,\?\:!;()«»…\]\[/\*–‽+&_\\½√>€™$•¼}{~—=“\"”″‟„]")
OOV_REGEX = "[^ '\-aiuenrbomkygwthszdcjfvplxq]"


DUMMY_CANDIDATES = [
    "a g k t t r k n a p r t f",
    "v w w x y x u r t g p w q",
    "n t r y t q q r u p t l n t",
    "p b r t u r e t f v w x u p z",
    "p p o j j k l n b f q t",
    "j k y u i t d s e w s r e j h i p p",
    "q w r e s f c t d r q g g y",
]


def replace_diacritics(text):
    text = re.sub(r"[éèëēêęěė]", "e", text)
    text = re.sub(r"[ãâāáäăâàąåạả]", "a", text)
    text = re.sub(r"[úūüùưûů]", "u", text)
    text = re.sub(r"[ôōóöõòő]", "o", text)
    text = re.sub(r"[ćçč]", "c", text)
    text = re.sub(r"[ïīíîıì]", "i", text)
    text = re.sub(r"[ñńňņ]", "n", text)
    text = re.sub(r"[țť]", "t", text)
    text = re.sub(r"[łľ]", "l", text)
    text = re.sub(r"[żžź]", "z", text)
    text = re.sub(r"[ğ]", "g", text)
    text = re.sub(r"[ř]", "r", text)
    text = re.sub(r"[ý]", "y", text)
    text = re.sub(r"[æ]", "ae", text)
    text = re.sub(r"[œ]", "oe", text)
    text = re.sub(r"[șşšś]", "s", text)
    return text


def preprocess_apostrophes_space_diacritics(text):
    text = APOSTROPHES_REGEX.sub("'", text) # replace different apostrophes by one
    text = re.sub(r"'+", "'", text)  # merge multiple apostrophes
    text = SPACE_REGEX.sub(" ", text) # replace different spaces by one
    text = replace_diacritics(text)

    text = re.sub(r" '", " ", text)  # delete apostrophes at the beginning of word
    text = re.sub(r"' ", " ", text)  # delete apostrophes at the end of word
    text = re.sub(r" +", " ", text)  # merge multiple spaces
    return text


def remove_oov_characters(text):
    text = re.sub(OOV_REGEX, " ", text)  # delete oov characters
    text = " ".join(text.split())
    return text


def get_title_and_text_from_json(content: str, exclude_titles: Set[str]) -> Tuple[str, str, str]:
    # Example of file content
    #   {"query":
    #     {"normalized":[{"from":"O'_Coffee_Club","to":"O' Coffee Club"}],
    #      "pages":
    #       {"49174116":
    #         {"pageid":49174116,
    #          "ns":0,
    #          "title":"O' Coffee Club",
    #          "extract":"O' Coffee Club (commonly known as Coffee Club) is a Singaporean coffee house..."
    #         }
    #       }
    #     }
    #   }
    try:
        js = json.loads(content.strip())
    except:
        print("cannot load json from text")
        return (None, None, None)
    if "query" not in js or "pages" not in js["query"]:
        print("no query[\"pages\"] in " + content)
        return (None, None, None)
    for page_key in js["query"]["pages"]:
        if page_key == "-1":
            continue
        page = js["query"]["pages"][page_key]
        if "title" not in page:
            continue
        title = page["title"]
        if title in exclude_titles:
            return (None, None, None)
        if "extract" not in page:
            continue
        text = page["extract"]
        title_clean = preprocess_apostrophes_space_diacritics(title)
        title_clean = CHARS_TO_IGNORE_REGEX.sub(" ", title_clean).lower()  # number of characters is the same in p and p_clean 
        return text, title, title_clean
    return (None, None, None)


def get_paragraphs_from_text(text):
    paragraphs = text.split("\n")
    for paragraph in paragraphs:
        if paragraph == "":
            continue
        p = preprocess_apostrophes_space_diacritics(paragraph)
        p_clean = CHARS_TO_IGNORE_REGEX.sub(" ", p).lower()  # number of characters is the same in p and p_clean 
        yield p, p_clean


def get_paragraphs_from_json(text, exclude_titles):
    # Example of file content
    #   {"query":
    #     {"normalized":[{"from":"O'_Coffee_Club","to":"O' Coffee Club"}],
    #      "pages":
    #       {"49174116":
    #         {"pageid":49174116,
    #          "ns":0,
    #          "title":"O' Coffee Club",
    #          "extract":"O' Coffee Club (commonly known as Coffee Club) is a Singaporean coffee house..."
    #         }
    #       }
    #     }
    #   }
    try:
        js = json.loads(text.strip())
    except:
        print("cannot load json from text")
        return
    if "query" not in js or "pages" not in js["query"]:
        print("no query[\"pages\"] in " + text)
        return
    for page_key in js["query"]["pages"]:
        if page_key == "-1":
            continue
        page = js["query"]["pages"][page_key]
        if "title" not in page:
            continue
        title = page["title"]
        if title in exclude_titles:
            continue
        if "extract" not in page:
            continue
        text = page["extract"]
        paragraphs = text.split("\n")
        for paragraph in paragraphs:
            if paragraph == "":
                continue
            p = preprocess_apostrophes_space_diacritics(paragraph)
            p_clean = CHARS_TO_IGNORE_REGEX.sub(" ", p).lower()  # number of characters is the same in p and p_clean 
            yield p, p_clean


def load_yago_entities(input_name: str, exclude_titles: Set[str]) -> Set[str]:
    yago_entities = set()
    with open(input_name, "r", encoding="utf-8") as f:
        for line in f:
            title_orig, title_clean = line.strip().split("\t")
            title_clean = title_clean.replace("_", " ")
            title_orig = title_orig.replace("_", " ")
            if title_orig in exclude_titles:
                print("skip: ", title_orig)
                continue
            yago_entities.add(title_clean)
    return yago_entities


def read_custom_phrases(filename: str, max_lines: int=-1, portion_size: int=-1) -> List[str]:
    """Reads custom phrases from input file.
    If input file contains multiple columns, only first column is used.
    """
    phrases = set()
    with open(filename, "r", encoding="utf-8") as f:
        n = 0
        n_for_portion = 0
        for line in f:
            parts = line.strip().split("\t")
            phrases.add(" ".join(list(parts[0].casefold().replace(" ", "_"))))
            if portion_size > 0 and n_for_portion >= portion_size:
                yield list(phrases)
                phrases = set()
                n_for_portion = 0
            if max_lines > 0 and n >= max_lines:
                yield list(phrases)
                return
            n += 1
            n_for_portion += 1
    yield list(phrases)


def load_ngram_mappings(input_name: str, max_dst_freq: int=1000000000) -> Tuple[defaultdict, Set]:
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
    max_len: int
) -> List[Tuple[str, str, float, float, int, int, int]]:
    hyp_letters = ["*"] + hyp_phrase.split()
    ref_letters = ["*"] + ref_phrase.split()
    DpInfo = namedtuple("DpInfo", ["hyp_pos", "ref_pos", "best_hyp_ngram_len", "best_ref_ngram_len", "score", "sum_score"])
    history = defaultdict(DpInfo)
    history[(0, 0)] = DpInfo(hyp_pos=0, ref_pos=0, best_hyp_ngram_len=1, best_ref_ngram_len=1, score=0.0, sum_score=0.0)
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
                hyp_ngram = " ".join(hyp_letters[(hyp_pos - hyp_ngram_len + 1):(hyp_pos+1)])
                for ref_ngram_len in range(1, 1 + min(max_len, ref_pos + 1)):
                    ref_ngram = " ".join(ref_letters[(ref_pos - ref_ngram_len + 1):(ref_pos+1)])
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

            assert(best_hyp_ngram_len > 0 or best_ref_ngram_len > 0)

            # save cell to history
            history[(hyp_pos, ref_pos)] = DpInfo(hyp_pos=hyp_pos, ref_pos=ref_pos, best_hyp_ngram_len=best_hyp_ngram_len, best_ref_ngram_len=best_ref_ngram_len, score=best_ngram_score, sum_score=best_sum_score)
    # now trace back on best path starting from last positions
    path = []
    hyp_pos = len(hyp_letters) - 1
    ref_pos = len(ref_letters) - 1
    cell_info = history[(hyp_pos, ref_pos)]
    path.append(cell_info)
    while hyp_pos > 0 and ref_pos > 0:
        hyp_pos -= cell_info.best_hyp_ngram_len
        ref_pos -= cell_info.best_ref_ngram_len 
        cell_info = history[(hyp_pos, ref_pos)]
        path.append(cell_info)

    result = []    
    for info in reversed(path):
        hyp_ngram = " ".join(hyp_letters[(info.hyp_pos-info.best_hyp_ngram_len+1):(info.hyp_pos+1)])
        ref_ngram = " ".join(ref_letters[(info.ref_pos-info.best_ref_ngram_len+1):(info.ref_pos+1)])
        joint_freq = joint_vocab.get((hyp_ngram, ref_ngram), 0)
        src_freq = src_vocab.get(hyp_ngram, 0)
        dst_freq = dst_vocab.get(ref_ngram, 0)
        result.append((hyp_ngram, ref_ngram, info.score, info.sum_score, joint_freq, src_freq, dst_freq))
    return result


def get_index(
    custom_phrases: List[str],
    vocab: defaultdict,
    ban_ngram_global: Set[str],
    min_log_prob: float=-4.0,
    max_phrases_per_ngram: int=100
) -> Tuple[List[str], Dict[str, int]]:
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
                        index_keys[b].update(new_ngrams) # = index_keys[b] | new_ngrams  #  join two dictionaries
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
    ngram2phrases = defaultdict(list)  # ngram to list of phrase ids

    for ngram in ngram_to_phrase_and_position:
        for phrase, b, length, lp in ngram_to_phrase_and_position[ngram]:
            if phrase not in phrase2id:
                phrases.append(phrase)
                phrase2id[phrase] = len(phrases) - 1
            ngram2phrases[ngram].append((phrase2id[phrase], b, length, lp))

    return phrases, ngram2phrases


def load_index(input_name: str) -> Tuple[List[str], Dict[str, int]]:
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


def search_in_index(ngram2phrases, phrases, letters):
    phrases2positions = np.zeros((len(phrases), len(letters)), dtype=float)
    position2ngrams = [set() for _ in range(len(letters))]  # positions mapped to sets of ngrams starting from that position

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


def get_token_list(text: str) -> List[str]:
    """Returns a list of tokens.

    This function expects that the tokens in the text are separated by space
    character(s). Example: "ca n't , touch".

    Args:
        text: String to be split into tokens.
    """
    return text.split()


def read_label_map(path: str) -> Dict[str, int]:
    """Return label map read from the given path."""
    with open(path, 'r') as f:
        label_map = {}
        empty_line_encountered = False
        for tag in f:
            tag = tag.strip()
            if tag:
                label_map[tag] = len(label_map)
            else:
                if empty_line_encountered:
                    raise ValueError('There should be no empty lines in the middle of the label map ' 'file.')
                empty_line_encountered = True
        return label_map


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


def get_candidates_with_most_coverage(
    phrases2positions: np.ndarray, phrase_lengths: List[int], max_candidates: int
) -> List[Tuple[float, int, int]]:
    """Returns k candidates whose ngrams cover most of the input text (compared to the candidate length).
       Args:
           phrases2positions: matrix where rows are phrases columns are letters of input sentence. Value is 1 on intersection of letter ngrams that were found in index leading to corresponding phrase. 
           phrase_lengths: list of phrase lengths (to avoid recalculation)
           max_candidates: required number of candidates
       Returns:
           List of tuples:
               coverage,
               approximate beginning position of the phrase
               phrase id
    """
    top = []
    for i in range(max_candidates): # add placeholders for best candidates
        heappush(top, (0.0, -1, -1))

    for i in range(len(phrase_lengths)):
        phrase_length = phrase_lengths[i]
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
        if coverage > top[0][0]:  # top[0] is the smallest element in the heap, top[0][0] - smallest coverage 
            heapreplace(top, (coverage, best_pos, i))
    return top


def get_candidates_with_most_coverage_on_whole_input(
    phrases2positions: np.ndarray, phrase_lengths: List[int], max_candidates: int
) -> List[Tuple[float, int, int]]:
    """Returns k candidates whose ngrams cover most of the input text (compared to the candidate length).
       Args:
           phrases2positions: matrix where rows are phrases columns are letters of input sentence. Value is 1 on intersection of letter ngrams that were found in index leading to corresponding phrase. 
           phrase_lengths: list of phrase lengths (to avoid recalculation)
           max_candidates: required number of candidates
       Returns:
           List of tuples:
               coverage,
               approximate beginning position of the phrase (in case of this function always 0)
               phrase id
    """
    top = []
    for i in range(max_candidates): # add placeholders for best candidates
        heappush(top, (0.0, -1, -1))

    coverage = np.sum(phrases2positions, axis=1) / (2 + phrases2positions.shape[1])
    indices = np.argpartition(coverage, -max_candidates)[-max_candidates:]

    for i in range(max_candidates):
        if coverage[indices[i]] >= 0.4:
            heapreplace(top, (coverage[indices[i]], 0, indices[i]))
    return top


def get_candidates(
    ngram2phrases: Dict[str, int],
    phrases: List[str],
    phrase_lengths: List[int],
    letters: List[str],
    max_candidates: int=10,
    min_real_coverage: float=0.8,
    add_dummy_candidates: bool=False,
    match_whole_input: bool=False
) -> List[str]:
    phrases2positions, position2ngrams = search_in_index(ngram2phrases, phrases, letters)
    if match_whole_input:
        top = get_candidates_with_most_coverage_on_whole_input(
            phrases2positions, phrase_lengths, 3 * max_candidates
        )
    else:
        top = get_candidates_with_most_coverage(
            phrases2positions, phrase_lengths, 3 * max_candidates
        )

    top_sorted = sorted(top, key=lambda item: item[0], reverse=True)
    # mask for each custom phrase, how many which symbols are covered by input ngrams
    phrases2coveredsymbols = [[0 for x in phrases[top_sorted[i][2]].split(" ")] for i in range(len(top_sorted))]
    candidates = []
    i = -1
    for coverage, begin, idx in top_sorted:
        i += 1
        phrase_length = phrase_lengths[idx]
        for pos in range(begin, begin + phrase_length):
            if pos >= len(position2ngrams):  # we do not know exact end of custom phrase in text, it can be different from phrase length
                break
            for ngram in position2ngrams[pos]:
                for phrase_id, b, size, lp in ngram2phrases[ngram]:
                    if phrase_id != idx:
                        continue
                    for ppos in range(b, b + size):
                        if ppos >= phrase_length:
                            break
                        phrases2coveredsymbols[i][ppos] = 1
                        
        real_coverage = sum(phrases2coveredsymbols[i]) / len(phrases2coveredsymbols[i])
        if real_coverage < min_real_coverage:
            continue
        candidates.append(phrases[idx])
        if len(candidates) >= max_candidates:
            break

    if add_dummy_candidates:
        while len(candidates) < max_candidates:
            candidates.append(random.choice(DUMMY_CANDIDATES))
    
    return candidates
