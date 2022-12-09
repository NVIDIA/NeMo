import math
import random
from argparse import ArgumentParser
from collections import defaultdict
from os.path import join
from typing import Dict, Optional, TextIO, Tuple

import numpy as np

## !!!this is temporary hack for my windows machine since is misses some installs 
sys.path.insert(1, "D:\\data\\work\\nemo\\nemo\\collections\\nlp\\data\\spellchecking_asr_customization")
print(sys.path)
from utils import get_all_candidates_coverage, get_index, load_ngram_mappings, search_in_index
# from nemo.collections.nlp.data.spellchecking_asr_customization.utils import get_all_candidates_coverage, get_index, load_ngram_mappings, search_in_index


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


vocab, ban_ngram = load_ngram_mappings(args.ngram_mapping, 10000)
custom_phrases = read_custom_vocab()
phrases, ngram2phrases = get_index(custom_phrases, vocab, ban_ngram)

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

path_parts = args.output_name.split("/")
path_parts2 = path_parts[-1].split(".")
doc_id = path_parts2[0]

out_debug = open(args.output_name + ".candidates", "w", encoding="utf-8")
out_debug2 = open(args.output_name + ".candidates_select", "w", encoding="utf-8")
out = open(args.output_name, "w", encoding="utf-8")
with open(args.input_file, "r", encoding="utf-8") as f:
    for line in f:
        short_sent, _ = line.strip().split("\t")
        sent = "_".join(short_sent.split())
        letters = list(sent)

        phrases2positions, position2ngrams = search_in_index(ngram2phrases, phrases, letters)
        candidate2coverage, candidate2position = get_all_candidates_coverage(phrases, phrases2positions)

        out_debug.write(" ".join(letters) + "\n")
        for pos in range(len(position2ngrams)):
            if len(position2ngrams[pos]) > 0:
                out_debug.write("\t\t" + str(pos) + "\t" + "|".join(list(position2ngrams[pos])) + "\n")

        # mask for each custom phrase, how many which symbols are covered by input ngrams
        phrases2coveredsymbols = [[0 for x in phrases[i].split(" ")] for i in range(len(phrases))]
        candidates = []
        k = 0
        for idx, coverage in sorted(enumerate(candidate2coverage), key=lambda item: item[1], reverse=True):
            begin = candidate2position[idx]  # this is most likely beginning of this candidate
            phrase_length = phrases[idx].count(" ") + 1
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
                            phrases2coveredsymbols[phrase_id][ppos] = 1
            k += 1
            if k > 20:
                break
            real_coverage = sum(phrases2coveredsymbols[idx]) / len(phrases2coveredsymbols[idx])
            if real_coverage < 0.8:
                out_debug.write("\t\t- " + phrases[idx] + "\tcov: " + str(coverage) + "\treal_cov: " + str(real_coverage) + "\n")
                continue
            candidates.append(phrases[idx])
            out_debug.write(
                "\t" + str(real_coverage) + "\t" + phrases[idx] + "\n" + " ".join(list(map(str, (map(int, phrases2positions[idx]))))) + "\n"
            )
            out_debug2.write(doc_id + "\t" + phrases[idx].replace(" ", "").replace("_", " ") + "\t" + short_sent + "\n")

        while len(candidates) < 10:
            candidates.append(random.choice(dummy_candidates))

        random.shuffle(candidates)
        if len(candidates) != 10:
            print("WARNING: cannot get 10 candidates", candidates)
            continue
        out.write(" ".join(letters) + "\t" + ";".join(candidates) + "\n")
out.close()
out_debug.close()
out_debug2.close()
