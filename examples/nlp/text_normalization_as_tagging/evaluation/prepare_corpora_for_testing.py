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
This script can be used to prepare test corpus for the ThutmoseTaggerModel from Google Text Normalization dataset.
"""

import os
import re
from argparse import ArgumentParser
from collections import Counter
from typing import Dict, TextIO, Tuple

from nemo.collections.nlp.data.text_normalization_as_tagging.utils import alpha_tokenize, spoken_preprocessing, written_preprocessing


parser = ArgumentParser(description="Text Normalization Data Preprocessing for English")
parser.add_argument(
    "--data_dir", required=True, type=str, help="Path to data directory with files like output-00000-of-00100.tsv"
)
parser.add_argument("--reference_vocab", required=True, type=str, help="Multi Reference vocabulary")
parser.add_argument("--output_file", required=True, type=str, help="Output file")
parser.add_argument("--lang", required=True, type=str, help="Language, e.g. en")
parser.add_argument(
    "--sampling_count", required=True, type=int, help="Number of examples per class, you want, use -1 for all examples"
)
parser.add_argument(
    "--tn_direction",
    action='store_true',
    help="Whether to run in TN direction, default is ITN",
)
args = parser.parse_args()


def process_file(
    inputname: str,
    out: TextIO,
    out_raw: TextIO,
    reference_vcb: Dict[Tuple[str, str], Dict[str, int]],
    sampling_vcb: Dict[str, int],
    lang: str,
) -> None:
    """Read one file from Google TN Dataset and extract test sentences

    Args:
        inputname: input file name
        out: test file in tsv-format, one line = one sentence
        out_raw: test file in Google TN Format (to be able to run duplex model on the same test set)
        reference_vcb: a vocabulary of multiple references (it is prepared beforehand)
        sampling_vcb: a Counter for different classes, used for sampling
        lang: language, e.g. "en" or "ru"

    Output line for ITN direction contains 3 columns:
    1. Spoken-domain input. Example:
         "this plan was enacted in nineteen eighty four and continued to be followed for nineteen years"
    2. Written-domain output. Example:
         "this plan was enacted in 1984 and continued to be followed for 19 years"
    3. Semiotic spans and reference translations. Coordinates in terms of input tokens (words). Example:
        DATE 5 8 | 1984/ | 1984 | nineteen eighty four | 1984,;CARDINAL 14 15 | xix | nineteen | 19: | 19

    Output line for TN direction contains 3 columns:
    1. Alpha-tokenized written-domain input. Example:
         "this plan was enacted in 1 9 8 4 and continued to be followed for 1 9 years"
    2. Spoken-domain output. Example:
         "this plan was enacted in nineteen eighty four and continued to be followed for nineteen years"
    3. Semiotic spans and reference translations. Coordinates in terms of input tokens. Example:
        DATE 5 9 | nineteen eighty four | one thousand eighty four;CARDINAL 15 17 | nineteen

    """
    input_tokens = []
    reference_fragments = []  # size may be different
    semiotic_info = []
    raw_lines = []
    sent_ok = True if args.sampling_count == -1 else False
    with open(inputname, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("<eos>"):
                if len(input_tokens) > 0 and sent_ok:
                    out.write(
                        " ".join(input_tokens) + "\t" + " ".join(reference_fragments) + "\t" + ";".join(semiotic_info) + "\n"
                    )
                    out_raw.write("\n".join(raw_lines) + "\n" + line)
                input_tokens = []
                reference_fragments = []
                semiotic_info = []
                raw_lines = []
                sent_ok = True if args.sampling_count == -1 else False
            else:
                raw_lines.append(line.strip())
                cls, written, spoken = line.strip().split("\t")
                spoken = spoken_preprocessing(spoken, lang)
                written = written_preprocessing(written, lang)
                references = set()
                if spoken == "sil":
                    continue
                if spoken == "<self>":
                    input_tokens.append(written)
                    reference_fragments.append(written)
                    if not args.tn_direction:
                        # if reference is <self>, but the word has itn conversions in our dictionary, add them
                        for cls in ["CARDINAL", "ORDINAL", "DATE"]:  # date, ex sixties -> 60s
                            k = (cls, written)
                            if k in reference_vcb:
                                for tr_variant in reference_vcb[k]:
                                    references.add(tr_variant)
                                semiotic_info.append(
                                    cls
                                    + " "
                                    + str(len(input_tokens) - 1)
                                    + " "
                                    + str(len(input_tokens))
                                    + " | "
                                    + " | ".join(references)
                                )
                                break
                    continue

                if args.tn_direction:  # TN direction
                    alpha_tokens = alpha_tokenize(written)
                    input_tokens.extend(alpha_tokens)
                    k = (cls, written)
                    if k in reference_vcb:
                        for tr_variant in reference_vcb[k]:
                            references.add(tr_variant)
                    references.add(spoken)
                    if cls == "PLAIN" or cls == "LETTERS":
                        references.add(written)  # there are cases in English corpus like colours/colors
                        reference_fragments.append(written)
                    else:
                        if cls == "TELEPHONE":  # correct google corpus issue
                            spoken = spoken.replace(" sil ", "")
                        reference_fragments.append(spoken)
                    semiotic_info.append(
                        cls
                        + " "
                        + str(len(input_tokens) - len(alpha_tokens))
                        + " "
                        + str(len(input_tokens))
                        + " | "
                        + " | ".join(list(references))
                    )

                else:  # ITN direction
                    spoken_words = spoken.split()
                    input_tokens.extend(spoken_words)

                    k = (cls, spoken)
                    if k in reference_vcb:
                        for tr_variant in reference_vcb[k]:
                            references.add(tr_variant)
                    references.add(spoken)
                    references.add(written)
                    for tr_variant in list(references):
                        # 6,51 km² => 6,51 km 2
                        (tr_variant2, n2) = re.subn(r"²", " 2", tr_variant)
                        (tr_variant3, n3) = re.subn(r"³", " 3", tr_variant)
                        if n2 > 0:
                            references.add(tr_variant2)
                        if n3 > 0:
                            references.add(tr_variant3)

                    semiotic_info.append(
                        cls
                        + " "
                        + str(len(input_tokens) - len(spoken_words))
                        + " "
                        + str(len(input_tokens))
                        + " | "
                        + " | ".join(list(references))
                    )
                    reference_fragments.append(written.casefold())

                if cls not in sampling_vcb:
                    sampling_vcb[cls] = 0
                if sampling_vcb[cls] < args.sampling_count:
                    sent_ok = True
                    sampling_vcb[cls] += 1


def main() -> None:
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data dir {args.data_dir} does not exist")
    reference_vcb = {}
    with open(args.reference_vocab, "r", encoding="utf-8") as f:
        for line in f:
            sem, spoken, written, freq = line.strip().split("\t")
            k = (sem, spoken)
            if k not in reference_vcb:
                reference_vcb[k] = {}
            reference_vcb[k][written] = int(freq)
    sampling_vcb = Counter()
    # test file in tsv-format, one line = one sentence
    out = open(args.output_file, "w", encoding="utf-8")
    # test file in Google TN Format (to be able to run duplex model on the same test set)
    out_raw = open(args.output_file + ".raw", "w", encoding="utf-8")
    print("out_raw=" + out_raw.name)
    input_paths = sorted([os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)])
    for inputname in input_paths:
        process_file(inputname, out, out_raw, reference_vcb, sampling_vcb, args.lang)
    out.close()
    out_raw.close()


if __name__ == "__main__":
    main()
