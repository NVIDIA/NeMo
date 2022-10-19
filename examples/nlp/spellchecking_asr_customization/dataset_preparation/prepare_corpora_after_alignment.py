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


"""
This script can be used to extract ngram mapping vocabulary from joined giza alignments and to index custom phrases.
"""

import glob
import os
import re
import math
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, Optional, TextIO, Tuple
from fuzzywuzzy import process

parser = ArgumentParser(description="Produce data for the Spellchecking ASR Customization")
parser.add_argument(
    "--mode",
    required=True,
    type=str,
    help='Mode, one of ["get_replacement_vocab", "index_by_vocab", "edit_distance"]',
)
parser.add_argument(
    "--alignment_filename", required=True, type=str, help='Name of alignment file, like "align.out"'
)
parser.add_argument("--out_filename", required=True, type=str, help='Output file')
parser.add_argument("--vocab_filename", required=True, type=str, help='Vocab name')
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


def get_replacement_vocab() -> None:
    """Loops through the file with alignment results, counts frequencies of different replacement segments.
    """

    full_vocab = defaultdict(dict)
    src_vocab = defaultdict(int)
    dst_vocab = defaultdict(int)
    n = 0
    with open(args.alignment_filename, "r", encoding="utf-8") as f:
        for line in f:
            n += 1
            if n % 100000 == 0:
                print(n)
            t = process_line(line)
            if t is None:
                continue
            src, dst, replacement = t
            inputs = src.split(" ")
            replacements = replacement.split(" ")
            if len(inputs) != len(replacements):
                raise ValueError("Length mismatch in: " + line)
            begin = 0
            for begin in range(len(inputs)):
                for end in range(begin + 1, begin + 5):
                    inp = " ".join(inputs[begin:end])
                    rep = " ".join(replacements[begin:end])
                    if not rep in full_vocab[inp]:
                        full_vocab[inp][rep] = 0
                    full_vocab[inp][rep] += 1
                    src_vocab[inp] += 1
                    dst_vocab[rep] += 1

    with open(args.vocab_filename, "w", encoding="utf-8") as out:
        for inp in full_vocab:
            for rep in full_vocab[inp]:
                out.write(inp + "\t" + rep + "\t" + str(full_vocab[inp][rep]) + "\t" + str(src_vocab[inp]) + "\t" + str(dst_vocab[rep]) + "\n")


def index_by_vocab() -> None:
    """Given a restricted vocabulary of replacements,
    loops through the file with custom phrases,
    generates all possible conversions and creates index.
    """

    if not os.path.exists(args.vocab_filename):
        raise ValueError(f"Vocab file {args.vocab_filename} does not exist")
    # load vocab from file
    vocab = defaultdict(dict)

    with open(args.vocab_filename, "r", encoding="utf-8") as f:
        for line in f:
            src, dst, joint_freq, src_freq, dst_freq = line.strip().split("\t")
            assert(src != "" and dst != ""), "src=" + src + "; dst=" + dst
            #-if dst.startswith("<DELETE>") or dst.endswith("<DELETE>"):
            #-    continue
            vocab[src][dst] = int(joint_freq) / int(src_freq)

    index_freq = defaultdict(int)
    ngram_to_phrase_and_position = defaultdict(list)
    ban_ngram = set()

    out = open(args.out_filename, "w", encoding="utf-8")
    with open(args.alignment_filename, "r", encoding="utf-8") as f:
        n = 0
        for line in f:
            n += 1
            if n % 1000 == 0:
                print (n)
            t = process_line(line)
            if t is None:
                continue
            phrase, _, _ = t
            #if not phrase.startswith("y o x t e r"):
            #    continue
            ok = True
            inputs = phrase.split(" ")
            begin = 0
            index_keys = [{} for i in inputs]  #key - letter ngram, index - beginning positions in phrase

            for begin in range(len(inputs)):
                for end in range(begin + 1, min(len(inputs) + 1, begin + 5)):
                    inp = " ".join(inputs[begin:end])
                    if inp not in vocab:
                        continue
                    for rep in vocab[inp]:
                        rep_before_clean = rep
                        lp = math.log(vocab[inp][rep])
                        rep = rep.replace("<DELETE>", "=")
                        if rep.strip() == "":
                            continue
                        for b in range(max(0, end - 5), end):  # try to grow previous ngrams with new replacement
                            new_ngrams = {}
                            for ngram in index_keys[b]:
                                lp_prev = index_keys[b][ngram]
                                if len(ngram) + len(rep) <= 10 and b + ngram.count(" ") == begin:
                                    if lp_prev + lp > -4.0:
                                        new_ngrams[ngram + rep + " "] = lp_prev + lp
                            index_keys[b] = index_keys[b] | new_ngrams   #  join two dictionaries
                        # add current replacement as ngram
                        if lp > -4.0:
                            index_keys[begin][rep + " "] = lp
                            #if phrase.startswith("y o x t e r"):
                            #    print(phrase, begin, rep, lp, rep_before_clean, inp, sep="; ")


            for b in range(len(index_keys)):
                for ngram, lp in sorted(index_keys[b].items(), key=lambda item: item[1], reverse=True):
                    real_length = ngram.count(" ")
                    ngram = ngram.replace("+", " ").replace("=", " ")
                    ngram = " ".join(ngram.split())
                    index_freq[ngram] += 1
                    if ngram in ban_ngram:
                        continue
                    ngram_to_phrase_and_position[ngram].append((phrase, b, real_length, lp))
                    if len(ngram_to_phrase_and_position[ngram]) > 100:
                        ban_ngram.add(ngram)
                        del ngram_to_phrase_and_position[ngram]
                        continue

    for ngram, freq in sorted(index_freq.items(), key=lambda item: item[1], reverse=True):
        for phrase, b, length, lp in ngram_to_phrase_and_position[ngram]:
            out.write(ngram + "\t" + phrase + "\t" + str(b) + "\t" + str(length) + "\t" + str(lp) + "\n")

    return
    correct = 0
    n = 0
    # now simulate search for pred_text of the same phrases
    with open(args.alignment_filename, "r", encoding="utf-8") as f:
        for line in f:
            n += 1
            if n % 1000 == 0:
                print (n)
            t = process_line(line)
            if t is None:
                continue
            ref, hyp, _ = t
            inputs = hyp.split(" ")
            candidates = defaultdict(int)
            begin = 0
            for begin in range(len(inputs)):
                for end in range(begin + 1, min(len(inputs) + 1, begin + 7)):
                    ngram = " ".join(inputs[begin:end]) + " "
                    if ngram not in ngram_to_phrase_and_position:
                        continue
                    for phrase, b, length, lp in ngram_to_phrase_and_position[ngram]:
                        candidates[phrase] += 1
            k = 0
            for candidate, freq in sorted(candidates.items(), key=lambda item: item[1], reverse=True):
                out.write(hyp + "\t" + candidate + "\t" + str(freq) + "\t" + ref + "\n")
                if candidate == ref:
                    correct += 1
                k += 1
                if k > 20:
                    break

    out.close()
    print("Correct=", correct)
    print("Total=", n)
    print("Accuracy=", correct/n)


def search_by_edit_distance():
    correct = 0
    n = 0
    refs = []
    hyps = []
    with open(args.alignment_filename, "r", encoding="utf-8") as f:
        for line in f:
            n += 1
            if n % 1000 == 0:
                print (n)
            t = process_line(line)
            if t is None:
                continue
            ref, hyp, _ = t
            ref = ref.replace(" ", "").replace("_", " ")
            hyp = hyp.replace(" ", "").replace("_", " ")
            refs.append(ref)
            hyps.append(hyp)

    for hyp, ref in zip(hyps, refs):
        candidates = process.extract(hyp, refs, limit=20)
        for candidate, score in candidates:
            print(hyp + "\t" + candidate + "\t" + str(score) + "\t" + ref)
            if candidate == ref:
                correct += 1
    print("Correct=", correct)
    print("Total=", n)
    print("Accuracy=", correct/n)


def main() -> None:
    if args.mode == "get_replacement_vocab":
        get_replacement_vocab()
    elif args.mode == "index_by_vocab":
        index_by_vocab()
    elif args.mode == "edit_distance":
        search_by_edit_distance()
    else:
        raise ValueError("unknown mode: " + args.mode)


if __name__ == "__main__":
    main()

