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


import os
import random
from argparse import ArgumentParser
from typing import List

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    get_all_candidates_coverage,
    get_index,
    load_ngram_mappings,
    search_in_index,
)

parser = ArgumentParser(description="Prepare input examples for inference")
parser.add_argument("--hypotheses_folder", required=True, type=str, help="Path to input folder with asr hypotheses")
parser.add_argument("--vocabs_folder", type=str, required=True, help="Path to input folder with user vocabs")
parser.add_argument("--output_folder", type=str, required=True, help="Output folder")
parser.add_argument("--ngram_mappings", type=str, required=True, help="Path to ngram mappings vocabulary")
parser.add_argument(
    "--sub_misspells_file",
    required=True,
    type=str,
    help="File with misspells from which only keys will be used to sample dummy candidates",
)
parser.add_argument("--debug", action='store_true', help="Whether to create files with debug information")


args = parser.parse_args()


def read_custom_vocab(filename: str) -> List[str]:
    phrases = set()
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            phrases.add(" ".join(list(line.strip().casefold().replace(" ", "_"))))
    return list(phrases)


print("load ngram mappings...")
ngram_mapping_vocab, ban_ngram = load_ngram_mappings(args.ngram_mappings, max_dst_freq=125000)
# CAUTION: entries in ban_ngram end with a space and can contain "+" "="
print("done.")

print("load big sample of phrases...")
big_sample_of_phrases = set()
with open(args.sub_misspells_file, "r", encoding="utf-8") as f:
    for line in f:
        phrase, _, _, src_freq, dst_freq = line.strip().split("\t")
        if int(src_freq) > 50:  # do not want to use frequent phrases as dummy candidates
            continue
        if len(phrase) < 6 or len(phrase) > 15:  # do not want to use too short or too long phrases as dummy candidates
            continue
        big_sample_of_phrases.add(phrase)

big_sample_of_phrases = list(big_sample_of_phrases)
print("done.")

print("load vocabs...")
custom_vocabs = {}  # key=doc_id, value=tuple(phrases, ngram2phrases)
for name in os.listdir(args.vocabs_folder):
    print("\tloading", name)
    parts = name.split(".")
    if len(parts) == 3 and parts[1] == "custom":
        doc_id = parts[0]
        custom_phrases = read_custom_vocab(os.path.join(args.vocabs_folder, name))
        if len(custom_phrases) == 0:
            continue
        phrases, ngram2phrases = get_index(custom_phrases, ngram_mapping_vocab, ban_ngram)
        custom_vocabs[doc_id] = (phrases, ngram2phrases)
print("done.")


for name in os.listdir(args.hypotheses_folder):
    parts = name.split(".")
    doc_id = parts[0]
    if doc_id not in custom_vocabs:
        continue
    phrases, ngram2phrases = custom_vocabs[doc_id]

    if args.debug:
        with open(os.path.join(args.output_folder, doc_id + ".index.txt"), "w", encoding="utf-8") as out_debug:
            for ngram in ngram2phrases:
                for phrase_id, b, size, lp in ngram2phrases[ngram]:
                    phr = phrases[phrase_id]
                    out_debug.write(ngram + "\t" + phr + "\t" + str(b) + "\t" + str(size) + "\t" + str(lp) + "\n")

    if args.debug:
        out_debug = open(os.path.join(args.output_folder, doc_id + ".candidates"), "w", encoding="utf-8")
        out_debug2 = open(os.path.join(args.output_folder, doc_id + ".candidates_select"), "w", encoding="utf-8")

    out = open(os.path.join(args.output_folder, doc_id + ".txt"), "w", encoding="utf-8")
    out_info = open(os.path.join(args.output_folder, doc_id + ".info.txt"), "w", encoding="utf-8")
    with open(os.path.join(args.hypotheses_folder, name), "r", encoding="utf-8") as f:
        for line in f:
            short_sent, _ = line.strip().split("\t")
            sent = "_".join(short_sent.split())
            letters = list(sent)

            phrases2positions, position2ngrams = search_in_index(ngram2phrases, phrases, letters)
            candidate2coverage, candidate2position = get_all_candidates_coverage(phrases, phrases2positions)

            if args.debug:
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
                if k > 20:
                    break
                real_coverage = sum(phrases2coveredsymbols[idx]) / len(phrases2coveredsymbols[idx])
                if real_coverage < 0.8:
                    if args.debug:
                        out_debug.write(
                            "\t\t- "
                            + phrases[idx]
                            + "\tcov: "
                            + str(coverage)
                            + "\treal_cov: "
                            + str(real_coverage)
                            + "\n"
                        )
                    continue
                candidates.append((phrases[idx], begin, phrase_length, coverage, real_coverage))
                if args.debug:
                    out_debug.write(
                        "\t"
                        + str(real_coverage)
                        + "\t"
                        + phrases[idx]
                        + "\n"
                        + " ".join(list(map(str, (map(int, phrases2positions[idx])))))
                        + "\n"
                    )
                    out_debug2.write(
                        doc_id + "\t" + phrases[idx].replace(" ", "").replace("_", " ") + "\t" + short_sent + "\n"
                    )

            # no need to process this short_sent further if it does not contain any real candidates
            if len(candidates) == 0:
                continue

            while len(candidates) < 10:
                dummy = random.choice(big_sample_of_phrases)
                dummy = " ".join(list(dummy.replace(" ", "_")))
                candidates.append((dummy, -1, dummy.count(" ") + 1, 0.0, 0.0))

            candidates = candidates[:10]
            random.shuffle(candidates)
            if len(candidates) != 10:
                print("WARNING: cannot get 10 candidates", candidates)
                continue
            out.write(" ".join(letters) + "\t" + ";".join([x[0] for x in candidates]) + "\n")
            info = ""
            for cand, begin, length, cov, real_cov in candidates:
                info += cand + "|" + str(begin) + "|" + str(length) + "|" + str(cov) + "|" + str(real_cov) + ";"
            out_info.write(info[:-1] + "\n")
    out.close()
    out_info.close()
    if args.debug:
        out_debug.close()
        out_debug2.close()
