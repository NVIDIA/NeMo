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


import json
import math
import random
import re
from collections import defaultdict, namedtuple
from typing import Dict, List, Set, Tuple, Union

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
    Input format: original \t misspelled \t joint_freq \t original_freq \t misspelled_freq
        u t o	u+i t o	49	8145	114
        u t o	<DELETE> t e	63	8145	16970
        u t o	o+_ t o	42	8145	1807
    """
    joint_vocab = defaultdict(int)
    orig_vocab = defaultdict(int)
    misspelled_vocab = defaultdict(int)
    max_len = 0
    with open(input_name, "r", encoding="utf-8") as f:
        for line in f:
            orig, misspelled, joint_freq, _, _ = line.strip().split("\t")
            if orig == "" or misspelled == "":
                raise ValueError("Emty n-gram: orig=" + orig + "; misspelled=" + misspelled)
            misspelled = misspelled.replace("<DELETE>", " ").replace("+", " ")
            misspelled = " ".join(misspelled.split())
            if misspelled == "":  # skip if resulting ngram doesn't contain any real character
                continue
            max_len = max(max_len, orig.count(" ") + 1, misspelled.count(" ") + 1)
            joint_vocab[(orig, misspelled)] += int(joint_freq)
            orig_vocab[orig] += int(joint_freq)
            misspelled_vocab[misspelled] += int(joint_freq)
    return joint_vocab, orig_vocab, misspelled_vocab, max_len


def get_alignment_by_dp(
    ref_phrase: str,
    hyp_phrase: str,
    dp_data: Tuple[defaultdict, defaultdict, defaultdict, int]
) -> List[Tuple[str, str, float, float, int, int, int]]:
    joint_vocab, orig_vocab, misspelled_vocab, max_len = dp_data
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
                    if (ref_ngram, hyp_ngram) not in joint_vocab:
                        continue
                    joint_freq = joint_vocab[(ref_ngram, hyp_ngram)]
                    orig_freq = orig_vocab.get(ref_ngram, 1)
                    ngram_score = math.log(joint_freq / orig_freq)
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
        joint_freq = joint_vocab.get((ref_ngram, hyp_ngram), 0)
        orig_freq = orig_vocab.get(ref_ngram, 0)
        misspelled_freq = misspelled_vocab.get(hyp_ngram, 0)
        result.append((hyp_ngram, ref_ngram, info.score, info.sum_score, joint_freq, orig_freq, misspelled_freq))
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

    Args:
        custom_phrases: list of all custom phrases, characters should be split by space,  real space replaced to underscore.
        vocab: n-gram mappings vocabulary,
        ban_ngram_global: set of banned n-grams,
        min_log_prob: minimum log probability, after which we stop growing this n-gram.
        max_phrases_per_ngram: maximum phrases that we allow to store per one n-gram. N-grams exceeding that quantity get banned.

    Returns:
        phrases - list of phrases. Position in this list is used as phrase_id.
        ngram2phrases - resulting index, i.e. dict where key=ngram, value=list of tuples (phrase_id, begin_pos, size, logprob)
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
    """ Load index from file
    """
    phrases = []  # id to phrase
    phrase2id = {}  # phrase to id
    ngram2phrases = defaultdict(list)  # ngram to list of tuples (phrase_id, begin_pos, size, logprob)
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
    ngram2phrases: Dict[str, List[Tuple[int, int, int, float]]], phrases: List[str], letters: Union[str, List[str]]
) -> Tuple[np.ndarray, List[Set[str]]]:
    """ Function used to search in index

    Args:
        ngram2phrases: dict where key=ngram, value=list of tuples (phrase_id, begin_pos, size, logprob)
        phrases: List of all phrases in custom vocabulary. Position corresponds to phrase_id.
        letters: list of letters of ASR-hypothesis. Should not contain spaces - real spaces should be replaced with underscores.

    Returns:
        phrases2positions: a matrix of size (len(phrases), len(letters)).
            It is filled with 1.0 (hits) on intersection of letter n-grams and phrases that are indexed by these n-grams, 0.0 - elsewhere.
            It is used later to find phrases with many hits within a contiguous window - potential matching candidates.
        position2ngrams: positions in ASR-hypothesis mapped to sets of ngrams starting from that position.
            It is used later to check how well each found candidate is covered by n-grams (to avoid cases where some repeating n-gram gives many hits to a phrase, but the phrase itself is not well covered).
    """

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
    """Get maximum hit coverage for each phrase - within a moving window of length of the phrase.
    Args:
        phrases: List of all phrases in custom vocabulary. Position corresponds to phrase_id.
        phrases2positions: a matrix of size (len(phrases), len(ASR-hypothesis)).
            It is filled with 1.0 (hits) on intersection of letter n-grams and phrases that are indexed by these n-grams, 0.0 - elsewhere.
    Returns:
        candidate2coverage: list of size len(phrases) containing coverage (0.0 to 1.0) in best window.
        candidate2position: list of size len(phrases) containing starting position of best window.
    """
    candidate2coverage = [0.0] * len(phrases)
    candidate2position = [-1] * len(phrases)

    for i in range(len(phrases)):
        phrase_length = phrases[i].count(" ") + 1
        all_coverage = np.sum(phrases2positions[i]) / phrase_length
        # if total coverage on whole ASR-hypothesis is too small, there is no sense in using moving window
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


def get_candidates(
    ngram2phrases: Dict[str, List[Tuple[int, int, int, float]]],
    phrases: List[str],
    letters: Union[str, List[str]],
    pool_for_random_candidates: List[str],
    min_phrase_coverage: float = 0.8,
) -> List[Tuple[str, int, int, float, float]]:
    """Given an index of custom vocabulary and an ASR-hypothesis retrieve 10 candidates.
    Args:
        ngram2phrases: dict where key=ngram, value=list of tuples (phrase_id, begin_pos, size, logprob)
        phrases: List of all phrases in custom vocabulary. Position corresponds to phrase_id.
        letters: list of letters of ASR-hypothesis. Should not contain spaces - real spaces should be replaced with underscores.
        pool_for_random_candidates: large list of strings, from which to sample random candidates in case when there are less than 10 real candidates
        min_phrase_coverage: We discard candidates which are not covered by n-grams to at least to this extent
          (to avoid cases where some repeating n-gram gives many hits to a phrase, but the phrase itself is not well covered).
     Returns:
        candidates: list of tuples (candidate_text, approximate_begin_position, length, coverage of window in ASR-hypothesis, coverage of phrase itself).
    """
    phrases2positions, position2ngrams = search_in_index(ngram2phrases, phrases, letters)
    candidate2coverage, candidate2position = get_all_candidates_coverage(phrases, phrases2positions)

    # mask for each custom phrase, how many which symbols are covered by input ngrams
    phrases2coveredsymbols = [[0 for x in phrases[i].split(" ")] for i in range(len(phrases))]
    candidates = []
    k = 0
    for idx, coverage in sorted(enumerate(candidate2coverage), key=lambda item: item[1], reverse=True):
        begin = candidate2position[idx]  # this is most likely beginning of this candidate
        phrase_length = phrases[idx].count(" ") + 1
        for pos in range(begin, begin + phrase_length):
            # we do not know exact end of custom phrase in text, it can be different from phrase length
            if pos >= len(position2ngrams):
                break
            for ngram in position2ngrams[pos]:
                for phrase_id, b, size, lp in ngram2phrases[ngram]:
                    if phrase_id != idx:
                        continue
                    for ppos in range(b, b + size):
                        if ppos >= phrase_length:
                            break
                        phrases2coveredsymbols[phrase_id][ppos] = 1
        k += 1
        if k > 100:
            break
        real_coverage = sum(phrases2coveredsymbols[idx]) / len(phrases2coveredsymbols[idx])
        if real_coverage < min_phrase_coverage:
            continue
        candidates.append((phrases[idx], begin, phrase_length, coverage, real_coverage))

    # no need to process this sentence further if it does not contain any real candidates
    if len(candidates) == 0:
        print("WARNING: no real candidates", candidates)
        return None

    while len(candidates) < 10:
        dummy = random.choice(pool_for_random_candidates)
        dummy = " ".join(list(dummy.replace(" ", "_")))
        candidates.append((dummy, -1, dummy.count(" ") + 1, 0.0, 0.0))

    candidates = candidates[:10]
    random.shuffle(candidates)
    if len(candidates) != 10:
        print("WARNING: cannot get 10 candidates", candidates)
        return None

    return candidates


def read_spellmapper_predictions(
    filename: str,
) -> List[Tuple[str, List[str], List[Tuple[int, int, str, float]], List[int]]]:
    # results is a list of (sent, list of candidates, list of fragment predictions, list of letter predictions)
    # fragment prediction is (begin, end, str, prob)
    results = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            text, candidate_str, fragment_predictions_str, letter_predictions_str = line.strip().split("\t")
            text = text.replace(" ", "").replace("_", " ")
            candidate_str = candidate_str.replace(" ", "").replace("_", " ")
            candidates = candidate_str.split(";")
            letter_predictions = list(map(int, letter_predictions_str.split()))
            if len(candidates) != 10:
                raise IndexError("expect 10 candidates, got: ", len(candidates))
            if len(text) != len(letter_predictions):
                raise IndexError("len(text)=", len(text), "; len(letter_predictions)=", len(letter_predictions))
            replacements = []
            if fragment_predictions_str != "":
                for prediction in fragment_predictions_str.split(";"):
                    begin, end, candidate_id, prob = prediction.split(" ")
                    begin = int(begin)
                    end = int(end)
                    candidate_id = int(candidate_id)
                    prob = float(prob)
                    replacements.append((begin, end, candidates[candidate_id - 1], prob))
                    replacements.sort()  # it will sort by begin, then by end
            results.append((text, replacements, letter_predictions))
    return results


def substitute_replacements_in_text(
    text: str, replacements: List[Tuple[int, int, str, float]], replace_hyphen_to_space: bool
) -> str:
    # Apply replacements to the input text, iterating from end to beginning, so that indexing does not change.
    # Note that we expect intersecting replacements to be filtered earlier.
    replacements.sort()
    last_begin = len(text) + 1
    corrected_text = text
    for begin, end, candidate, prob in reversed(replacements):
        if end > last_begin:
            print("WARNING: skip intersecting replacement [", candidate, "] in text: ", text)
            continue
        if replace_hyphen_to_space:
            candidate = candidate.replace("-", " ")
        corrected_text = corrected_text[:begin] + candidate + corrected_text[end:]
        last_begin = begin
    return corrected_text


def apply_replacements_to_text(
    text: str,
    replacements: List[Tuple[int, int, str, float]],
    min_prob: float = 0.5,
    replace_hyphen_to_space = False,
    dp_data: Tuple[defaultdict, defaultdict, defaultdict, int] = None,
    min_dp_score_per_symbol: float = -99.9
):
    # sort replacements by positions
    replacements.sort()
    # filter replacements
    # Note that we do not skip replacements with same text, otherwise intersecting candidates with lower probability can win
    filtered_replacements = []
    for j in range(len(replacements)):
        replacement = replacements[j]
        begin, end, candidate, prob = replacement
        fragment = text[begin:end]
        candidate_spaced = " ".join(list(candidate.replace(" ", "_")))
        fragment_spaced = " ".join(list(fragment.replace(" ", "_")))
        # apply penalty if candidate length is bigger than fragment length
        # to avoid cases like "forward-looking" replacing "looking" in "forward looking" resulting in "forward forward looking"
        if len(candidate) > len(fragment):
            penalty = len(fragment) / len(candidate)
            prob *= penalty 
        # skip replacement with low probability
        if prob < min_prob:
            continue
        # skip replacements with some predefined templates, e.g. "*'s" => "*s"
        if check_banned_replacements(fragment, candidate):
            continue
        if dp_data is not None:
            path = get_alignment_by_dp(candidate_spaced, fragment_spaced, dp_data)
            # path[-1][3] is the sum of logprobs for best path of dynamic programming: divide sum_score by length
            if path[-1][3] / (len(fragment)) < min_dp_score_per_symbol:
                continue

        # skip replacement if it intersects with previous replacement and has lower probability, otherwise remove previous replacement
        if len(filtered_replacements) > 0 and filtered_replacements[-1][1] > begin:
            if filtered_replacements[-1][3] > prob:
                continue
            else:
                filtered_replacements.pop()
        filtered_replacements.append((begin, end, candidate, prob))

    return substitute_replacements_in_text(text, filtered_replacements, replace_hyphen_to_space)


def update_json_with_spellmapper_corrections(
    input_name: str,
    output_name: str,
    spellmapper_results: List[Tuple[str, List[str], List[Tuple[int, int, int, float]],List[int]]],
    min_prob: float = 0.5,
    replace_hyphen_to_space=True,
) -> None:
    out = open(output_name, "w", encoding="utf-8")
    input_lines = []
    with open(input_name, "r", encoding="utf-8") as f:
        input_lines = f.readlines()
    if len(input_lines) != len(spellmapper_results):
        raise IndexError(
            "len(input_lines)=", len(input_lines), "; len(spellmapper_results)=", len(spellmapper_results)
        )
    for i in range(len(input_lines)):
        text, replacements, _ = spellmapper_results[i]
        data = json.loads(input_lines[i].strip())
        if text != data["pred_text"]:
            raise IndexError("Line mismatch: text=", text, "data[\"pred_text\"]", data["pred_text"])
        # store old predicted text in another field
        data["pred_text_before_correction"] = data["pred_text"]
        data["pred_text"] = apply_replacements_to_text(
            text, replacements, min_prob=min_prob, replace_hyphen_to_space=replace_hyphen_to_space
        )
        out.write(json.dumps(data) + "\n")
    out.close()


def check_banned_replacements(src, dst):
    # customers' => customer's
    if src.endswith("s'") and dst.endswith("'s") and src[0:-2] == dst[0:-2]:
        return True
    # customer's => customers'
    if src.endswith("'s") and dst.endswith("s'") and src[0:-2] == dst[0:-2]:
        return True
    # customers => customer's
    if src.endswith("s") and dst.endswith("'s") and src[0:-1] == dst[0:-2]:
        return True
    # customer's => customers
    if src.endswith("'s") and dst.endswith("s") and src[0:-2] == dst[0:-1]:
        return True
    # customers => customers'
    if src.endswith("s") and dst.endswith("s'") and src[0:-1] == dst[0:-2]:
        return True
    # customers' => customers
    if src.endswith("s'") and dst.endswith("s") and src[0:-2] == dst[0:-1]:
        return True
    # utilities => utility's
    if src.endswith("ies") and dst.endswith("y's") and src[0:-3] == dst[0:-3]:
        return True
    # utility's => utilities
    if src.endswith("y's") and dst.endswith("ies") and src[0:-3] == dst[0:-3]:
        return True
    # utilities => utility
    if src.endswith("ies") and dst.endswith("y") and src[0:-3] == dst[0:-1]:
        return True
    # utility => utilities
    if src.endswith("y") and dst.endswith("ies") and src[0:-1] == dst[0:-3]:
        return True
    # group is => group's
    if src.endswith(" is") and dst.endswith("'s") and src[0:-3] == dst[0:-2]:
        return True
    # group's => group is
    if src.endswith("'s") and dst.endswith(" is") and src[0:-2] == dst[0:-3]:
        return True
    # trex's => trex
    if src.endswith("'s") and src[0:-2] == dst:
        return True
    # trex => trex's
    if dst.endswith("'s") and dst[0:-2] == src:
        return True
    # increases => increase (but trimass => trimas is ok)
    if src.endswith("s") and (not src.endswith("ss")) and src[0:-1] == dst:
        return True
    # increase => increases ((but trimas => trimass is ok))
    if dst.endswith("s") and (not dst.endswith("ss")) and dst[0:-1] == src:
        return True
    # anticipate => anticipated
    if src.endswith("e") and dst.endswith("ed") and src[0:-1] == dst[0:-2]:
        return True
    # anticipated => anticipate
    if src.endswith("ed") and dst.endswith("e") and src[0:-2] == dst[0:-1]:
        return True
    # regarded => regard
    if src.endswith("ed") and src[0:-2] == dst:
        return True
    # regard => regarded
    if dst.endswith("ed") and dst[0:-2] == src:
        return True
    # longer => long
    if src.endswith("er") and src[0:-2] == dst:
        return True
    # long => longer
    if dst.endswith("er") and dst[0:-2] == src:
        return True
    # discussed => discussing
    if src.endswith("ed") and dst.endswith("ing") and src[0:-2] == dst[0:-3]:
        return True
    # discussing => discussed
    if src.endswith("ing") and dst.endswith("ed") and src[0:-3] == dst[0:-2]:
        return True
    # discussion => discussing
    if src.endswith("ion") and dst.endswith("ing") and src[0:-3] == dst[0:-3]:
        return True
    # discussing => discussion
    if src.endswith("ing") and dst.endswith("ion") and src[0:-3] == dst[0:-3]:
        return True
    # dispensers => dispensing
    if src.endswith("ers") and dst.endswith("ing") and src[0:-3] == dst[0:-3]:
        return True
    # dispensing => dispensers
    if src.endswith("ing") and dst.endswith("ers") and src[0:-3] == dst[0:-3]:
        return True
    # discussion => discussed
    if src.endswith("ion") and dst.endswith("ed") and src[0:-3] == dst[0:-2]:
        return True
    # discussed => discussion
    if src.endswith("ed") and dst.endswith("ion") and src[0:-2] == dst[0:-3]:
        return True
    # incremental => increment
    if src.endswith("ntal") and dst.endswith("nt") and src[0:-4] == dst[0:-2]:
        return True
    # increment => incremental
    if src.endswith("nt") and dst.endswith("ntal") and src[0:-2] == dst[0:-4]:
        return True
    # delivery => deliverer
    if src.endswith("ery") and dst.endswith("erer") and src[0:-3] == dst[0:-4]:
        return True
    # deliverer => delivery
    if src.endswith("erer") and dst.endswith("ery") and src[0:-4] == dst[0:-3]:
        return True
    # comparably => comparable
    if src.endswith("bly") and dst.endswith("ble") and src[0:-3] == dst[0:-3]:
        return True
    # comparable => comparably
    if src.endswith("ble") and dst.endswith("bly") and src[0:-3] == dst[0:-3]:
        return True
    # beautiful => beautifully
    if src.endswith("l") and dst.endswith("lly") and src[0:-1] == dst[0:-3]:
        return True
    # beautifully => beautiful
    if src.endswith("lly") and dst.endswith("l") and src[0:-3] == dst[0:-1]:
        return True
    # america => american
    if src.endswith("a") and dst.endswith("an") and src[0:-1] == dst[0:-2]:
        return True
    # american => america
    if src.endswith("an") and dst.endswith("a") and src[0:-2] == dst[0:-1]:
        return True
    # reinvesting => investing
    if src.startswith("re") and src[2:] == dst:
        return True
    # investing => reinvesting
    if dst.startswith("re") and dst[2:] == src:
        return True
    # outperformance => performance
    if src.startswith("out") and src[3:] == dst:
        return True
    # performance => outperformance
    if dst.startswith("out") and dst[3:] == src:
        return True
    return False
