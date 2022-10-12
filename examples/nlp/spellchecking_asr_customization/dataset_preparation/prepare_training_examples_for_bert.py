import json
import random
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from os.path import join
from argparse import ArgumentParser
from typing import Dict, Optional, TextIO, Tuple
from numba import jit

parser = ArgumentParser(
    description="Prepare training examples for Bert: insert custom phrases and best candidates into sample sentences"
)
parser.add_argument("--input_manifest", required=True, type=str, help="Path to manifest file with sample sentences")
parser.add_argument("--input_vocab", type=str, required=True, help="Path to simulated custom vocabulary")
parser.add_argument("--index_name", required=True, type=str, help="Path to file with index of custom phrases")
parser.add_argument("--output_name", type=str, required=True, help="Output file")

args = parser.parse_args()

def process_line(line: str) -> Optional[Tuple[str, str, str, int]]:
    """A helper function to read the file with alignment results"""

    parts = line.strip().split("\t")
    if len(parts) != 4:
        return None
    if parts[0] != "good:":
        return None

    src, dst, align = parts[1], parts[2], parts[3]

    return src, dst, align


def read_manifest(path):
    manifest = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest


def read_custom_vocab():
    refs = []
    hyps = []
    with open(args.input_vocab, "r", encoding="utf-8") as f:
        for line in f:
            t = process_line(line)
            if t is None:
                continue
            ref, hyp, _ = t
            refs.append(ref)
            hyps.append(hyp)
    return refs, hyps


def read_index():
    phrases = []   # id to phrase
    phrase2id = {}  # phrase to id
    ngram2phrases = defaultdict(list) # ngram to list of phrase ids
    with open(args.index_name, "r", encoding="utf-8") as f:
        for line in f:
            ngram, phrase, begin, length, lp = line.strip().split("\t")
            if phrase not in phrase2id:
                phrases.append(phrase)
                phrase2id[phrase] = len(phrases) - 1
            ngram2phrases[ngram].append((phrase2id[phrase], int(begin), int(length), float(lp)))
    return phrases, ngram2phrases


@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
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

        coverage = max_sum / (phrase_length + 2)    # smoothing
        candidate2coverage[i] = coverage
        candidate2position[i] = best_pos
    return candidate2coverage, candidate2position


refs, hyps = read_custom_vocab()
manifest_data = read_manifest(args.input_manifest)

# extract just the text corpus from the manifest
text = [data['text'] for data in manifest_data]

phrases, ngram2phrases = read_index()
print("len(phrases)=", len(phrases), "; len(ngram2phrases)=", len(ngram2phrases))

correct = 0  # debug counter for cases when correct candidate was in top10
with open(args.output_name, "w", encoding="utf-8") as out:
    for i in range(len(refs)):    # loop through custom phrases
        p = random.randrange(len(text))  # pick random sentence
        sent = text[p]
        words = sent.split()
        if len(words) > 15:
            s = random.randrange(len(words) - 15)
            words = words[s:s+15]

        # choose random position to insert custom phrase
        r = random.randrange(len(words))
        sent_begin = "_".join(words[0:r])
        sent_end = "_".join(words[r:])

        sent_begin_letters = list(sent_begin)
        if len(sent_begin_letters) > 0:
            sent_begin_letters.append("_")

        sent_end_letters = list(sent_end)
        if len(sent_end_letters) > 0:
            sent_end_letters = ["_"] + sent_end_letters

        final_sent = " ".join(sent_begin_letters) + " " + hyps[i] + " " + " ".join(sent_end_letters)
        final_sent = final_sent.strip()
        hyp_len = len(hyps[i].split(" "))
        begin_len = len(sent_begin_letters)

        reference, position, length = refs[i], begin_len, hyp_len
        letters = final_sent.split()

        phrases2positions = np.zeros((len(phrases), len(letters)), dtype=float)
        position2ngrams = [{}] * len(letters)   # positions mapped to dicts of ngrams starting from that position

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
        for idx, coverage in sorted(enumerate(candidate2coverage), key=lambda item: item[1], reverse=True):
            k += 1
            if k > 10:
                break
            candidates.append(phrases[idx])

        random.shuffle(candidates)
        for k in range(len(candidates)):
            if candidates[k] == reference:
                correct += 1
                correct_id = k + 1    # correct index is 1-based
        if len(candidates) != 10:
            print(final_sent)
            print("WARNING: cannot get 10 candidates", candidates)
            continue
        out.write(final_sent + ";" + ";".join(candidates) + "\t" + str(correct_id) + "\tCUSTOM " + str(begin_len) + " " + str(begin_len + hyp_len) + "\n")
print ("Correct=", correct)