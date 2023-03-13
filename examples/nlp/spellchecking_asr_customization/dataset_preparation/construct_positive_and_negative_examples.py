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


"""
This script contains logics for construction of training examples from Yago Wikipedia preprocessed data.

Negative example consists of 2 columns:
    original text fragment
    10 candidates

Positive example consists of 5 columns:
    original text fragment
    10 candidates
    targets ids
    target spans
    target misspells
Example:
    ninjas knuckle up and three ninjas kick back as	
    *beckjay;&jean-louis bonenfant;knuckle;*layback spin;*ninjatown;&antecedent;*grabocka;&ashrita;*licinius;ninjas
    3 10 10
    7 14;0 6;28 34
    knockle;ninjs;ninjas

"""

import random
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List, Set, TextIO

parser = ArgumentParser(description="Preparation of training examples from Yago Wikipedia preprocessed data")

# evantra was presented	1	[evantra] 0 7
# at the twenty thirteen top marques monaco show it was	1 2 3	[top marques monaco] 23 41;[marques] 27 34;[monaco] 35 41
parser.add_argument(
    "--non_empty_fragments_file",
    required=True,
    type=str,
    help="Input file with fragments containing at least 1 target phrase",
)

# of singapore began airing	0
# singapore began airing on channel twelve on the	0
parser.add_argument("--empty_fragments_file", required=True, type=str, help="Input file with empty fragments")

# berlin_conclusion       beyond_inclusion        8
parser.add_argument(
    "--related_phrases_file",
    required=True,
    type=str,
    help="File with related phrases used to sample some similar candidates",
)

# auto    otto    38      314     2332
# auto    auto    239     314     824
# awf tag-team championship       off tag team championship       1       1       1
parser.add_argument(
    "--sub_misspells_file",
    required=True,
    type=str,
    help="File with misspells used to sample misspells for target phrases",
)

# muhammad	mohamad;mohammad;muhammed;muhammad's;mohammed;amadeus;l'homme;hamad;mohamed
# johann	joanna;gianni;joanne;johannes;johanna;joined;johan;jahan;jeanne
# garwolin railway station	station is a railway station;railway station is a railway
# dan avidan	is a village and a
parser.add_argument(
    "--reverse_frequent_ngrams_candidates", required=True, type=str, help="File with potential false positives"
)

parser.add_argument(
    "--output_file_positive",
    required=True,
    type=str,
    help="Output file for examples with at least 1 correct candidate",
)
parser.add_argument(
    "--output_file_negative", required=True, type=str, help="Output file for examples with no correct candidates"
)
args = parser.parse_args()


def make_negative_example(
    text: str, false_positive_vocab: Dict[str, Set[str]], big_sample_of_phrases: List[str], out: TextIO
) -> None:
    incorrect_phrases = set()
    words = text.split()
    ngrams = set()

    for begin in range(len(words)):
        for end in range(begin + 1, min(begin + 5, len(words) + 1)):
            ngram = " ".join(words[begin:end])
            ngrams.add(ngram)

    for ngram in ngrams:
        if ngram in false_positive_vocab:
            phrases = []
            for phrase in false_positive_vocab[ngram]:
                # skip candidates that are present as ngrams in this fragment (they should not appear as incorrect candidates)
                if phrase in ngrams:
                    continue
                phrases.append(phrase)
            if len(phrases) == 0:
                continue
            for phr in random.choices(phrases, k=3):
                incorrect_phrases.add("*" + phr)

    while len(incorrect_phrases) < 10:
        cand = random.choice(big_sample_of_phrases)  # take just random phrase as candidate
        if cand in ngrams:
            continue
        incorrect_phrases.add("&" + cand)

    incorrect_phrases = list(incorrect_phrases)
    random.shuffle(incorrect_phrases)
    incorrect_phrases = incorrect_phrases[:10]

    out.write(text + "\t" + ";".join(incorrect_phrases) + "\n")


def process_negative_examples(
    filename: str, out_negative: TextIO, false_positive_vocab: Dict[str, Set[str]], big_sample_of_phrases: List[str]
) -> None:
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            # singapore began airing on channel twelve on the	0
            text, _ = line.strip().split("\t")
            make_negative_example(text, false_positive_vocab, big_sample_of_phrases, out_negative)


def process_positive_examples(
    filename: str,
    out_positive: TextIO,
    out_negative: TextIO,
    misspells_vocab: Dict[str, Dict[str, float]],
    related_vocab: Dict[str, Dict[str, int]],
    false_positive_vocab: Dict[str, Set[str]],
    big_sample_of_phrases: List[str],
) -> None:
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            # at the twenty thirteen top marques monaco show it was	1 2 3	[top marques monaco] 23 41;[marques] 27 34;[monaco] 35 41
            text, _, span_str = line.strip().split("\t")
            span_infos = span_str.split(";")
            random.shuffle(span_infos)
            # here we store positions already covered by some correct candidates (as intersection of correct candidates is not allowed)
            mask = [0] * len(text)
            selected = defaultdict(list)  # key=phrase, value=list of tuples (begin, end)
            skipped = set()
            for span_info in span_infos:
                phrase, position_str = span_info[1:].split("] ")  # 1: to skip first bracket
                begin, end = position_str.split(" ")
                begin = int(begin)
                end = int(end)
                # if this is duplicate phrase in one sentence and it was already skipped, skip again
                if phrase in skipped:
                    continue
                # if this is duplicate phrase in one sentence and it was already selected, do not skip
                if phrase not in selected:
                    if random.uniform(0, 1) > 0.9:  # skip candidate with probability 0.1
                        skipped.add(phrase)
                        continue
                    # skip candidate if it intersects with positions already occupied by another candidate
                    if any(mask[begin:end]):
                        skipped.add(phrase)
                        continue
                mask[begin:end] = [1] * (end - begin)
                selected[phrase].append((begin, end))
            if len(selected) == 0:  # if no correct candidates were selected make this a negative example
                make_negative_example(text, false_positive_vocab, big_sample_of_phrases, out_negative)
                continue

            words = text.split()
            ngrams = set()

            for begin in range(len(words)):
                for end in range(begin + 1, min(begin + 5, len(words) + 1)):
                    ngram = " ".join(words[begin:end])
                    ngrams.add(ngram)

            incorrect_phrases1 = set()  # here we store incorrect candidates from source 1: related phrases

            # add phrases "related" to correct candidates
            for phrase in selected:
                if phrase in related_vocab:
                    phrases = []
                    weights = []
                    for related_phrase in related_vocab[phrase]:
                        if related_phrase in ngrams:
                            continue
                        phrases.append(related_phrase)
                        weights.append(related_vocab[phrase][related_phrase])
                    if len(phrases) == 0:
                        continue
                    # take 3 or less (if same choice occur)
                    for phr in random.choices(phrases, weights=weights, k=min(3, len(phrases))):
                        incorrect_phrases1.add("#" + phr)

            incorrect_phrases2 = set()  # here we store incorrect candidates from source 2: false positives

            for ngram in ngrams:
                if ngram in false_positive_vocab:
                    phrases = []
                    for phrase in false_positive_vocab[ngram]:
                        if phrase in ngrams:
                            continue
                        phrases.append(phrase)
                    if len(phrases) == 0:
                        continue
                    for phr in random.choices(phrases, k=min(3, len(phrases))):
                        incorrect_phrases2.add("*" + phr)

            selected_incorrect_candidates = set()
            if len(incorrect_phrases1) > 0:
                selected_incorrect_candidates.update(set(random.choices(list(incorrect_phrases1), k=5)))
            if len(incorrect_phrases2) > 0:
                selected_incorrect_candidates.update(set(random.choices(list(incorrect_phrases2), k=5)))

            needed_incorrect_count = 10 - len(selected)
            while len(selected_incorrect_candidates) < needed_incorrect_count:
                cand = random.choice(big_sample_of_phrases)  # take just random phrase as candidate
                if cand in ngrams:
                    continue
                selected_incorrect_candidates.add("&" + cand)

            selected_incorrect_candidates = list(selected_incorrect_candidates)
            random.shuffle(selected_incorrect_candidates)

            candidates = list(selected.keys()) + selected_incorrect_candidates[:needed_incorrect_count]
            random.shuffle(candidates)

            targets = []
            spans = []
            misspelled_targets = []
            for idx, cand in enumerate(candidates):
                if cand in selected:
                    for begin, end in selected[cand]:
                        targets.append(str(idx + 1))  # targets are 1-based
                        spans.append(str(begin) + " " + str(end))
                        misspells = []
                        weights = []
                        misspells.append(cand)  # no misspell variant
                        weights.append(1)
                        sum = 1.0  # smoothing
                        for misspell in misspells_vocab[cand]:
                            sum += misspells_vocab[cand][misspell]
                            misspells.append(misspell)
                            weights.append(misspells_vocab[cand][misspell])
                        weights[0] = 0.1 * sum  #  no misspell variant will be chosen in 10%
                        misspell = random.choices(misspells, weights=weights, k=1)
                        misspelled_targets.append(misspell[0])  # this list has a single element

            out_positive.write(
                text
                + "\t"
                + ";".join(candidates)
                + "\t"
                + " ".join(targets)
                + "\t"
                + ";".join(spans)
                + "\t"
                + ";".join(misspelled_targets)
                + "\n"
            )


def main() -> None:
    misspells_vocab = defaultdict(dict)
    with open(args.sub_misspells_file, "r", encoding="utf-8") as f:
        for line in f:
            phrase, misspelled_phrase, joint_freq, src_freq, dst_freq = line.strip().split("\t")
            if phrase == misspelled_phrase:
                continue
            if misspelled_phrase != misspelled_phrase.strip():  # bug fix, found a phrase with space at the end
                continue
            misspells_vocab[phrase][misspelled_phrase] = int(joint_freq)

    related_vocab = defaultdict(dict)
    with open(args.related_phrases_file, "r", encoding="utf-8") as f:
        for line in f:
            phrase, related_phrase, covered_symbols = line.strip().split("\t")
            if phrase == related_phrase:
                continue
            phrase = phrase.replace("_", " ")
            related_phrase = related_phrase.replace("_", " ")
            related_vocab[phrase][related_phrase] = int(covered_symbols)

    false_positive_vocab = defaultdict(set)
    with open(args.reverse_frequent_ngrams_candidates, "r", encoding="utf-8") as f:
        for line in f:
            phrase, inputs = line.strip().split("\t")
            for inp in inputs.split(";"):
                if inp == phrase:
                    continue
                if inp not in false_positive_vocab:
                    false_positive_vocab[inp] = set()
                false_positive_vocab[inp].add(phrase)

    big_sample_of_phrases = list(misspells_vocab.keys())

    random.seed(0)

    out_positive = open(args.output_file_positive, "w", encoding="utf-8")
    out_negative = open(args.output_file_negative, "w", encoding="utf-8")
    process_positive_examples(
        args.non_empty_fragments_file,
        out_positive,
        out_negative,
        misspells_vocab,
        related_vocab,
        false_positive_vocab,
        big_sample_of_phrases,
    )
    process_negative_examples(args.empty_fragments_file, out_negative, false_positive_vocab, big_sample_of_phrases)
    out_positive.close()
    out_negative.close()


if __name__ == "__main__":
    main()
