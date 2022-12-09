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


def get_candidates(phrases, letters):
    phrases2positions, position2ngrams = search_in_index(ngram2phrases, phrases, letters)
    candidate2coverage, candidate2position = get_all_candidates_coverage(phrases, phrases2positions)

    candidates = []
    k = 0
    for idx, coverage in sorted(enumerate(candidate2coverage), key=lambda item: item[1], reverse=True):
        k += 1
        if k > 10:
            break
        candidates.append(phrases[idx])
    return candidates


refs, hyps = read_custom_vocab()
manifest_data = read_manifest(args.input_manifest)

# extract just the text corpus from the manifest
text = [data['text'] for data in manifest_data]

phrases, ngram2phrases = load_index(args.index_name)
print("len(phrases)=", len(phrases), "; len(ngram2phrases)=", len(ngram2phrases))

correct = 0  # debug counter for cases when correct candidate was in top10
with open(args.output_name, "w", encoding="utf-8") as out:
    # mostly positive examples
    for i in range(len(refs)):  # loop through custom phrases
        p = random.randrange(len(text))  # pick random sentence
        sent = text[p]
        words = sent.split()
        if len(words) > 10:
            s = random.randrange(len(words) - 10)
            words = words[s : s + 10]

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

        candidates = get_candidates(phrases, letters)

        random.shuffle(candidates)
        correct_id = 0
        for k in range(len(candidates)):
            if candidates[k] == reference:
                correct += 1
                correct_id = k + 1  # correct index is 1-based
        if len(candidates) != 10:
            print(final_sent)
            print("WARNING: cannot get 10 candidates", candidates)
            continue
        out.write(
            final_sent
            + "\t"
            + ";".join(candidates)
            + "\t"
            + str(correct_id)
            + "\tCUSTOM "
            + str(begin_len)
            + " "
            + str(begin_len + hyp_len)
            + "\n"
        )
    # add negative examples (no correct candidate)
    for i in range(len(refs)):  # loop through custom phrases
        p = random.randrange(len(text))  # pick random sentence
        sent = text[p]
        words = sent.split()
        if len(words) > 10:
            s = random.randrange(len(words) - 10)
            words = words[s : s + 10]

        sent_letters = list("_".join(words))

        candidates = get_candidates(phrases, sent_letters)

        random.shuffle(candidates)
        correct_id = 0
        out.write(" ".join(sent_letters) + "\t" + ";".join(candidates) + "\t0\t\n")
print("Correct=", correct)
