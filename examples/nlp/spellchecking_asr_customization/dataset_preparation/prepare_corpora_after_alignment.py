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
This script can be used to 1) extract alignments from GIZA++ output, 2) build n-gram mapping vocabulary and 3) extract aligned subphrases.
"""

from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from nemo.collections.nlp.data.text_normalization_as_tagging.utils import (
    check_monotonicity,
    fill_alignment_matrix,
    get_targets,
)


parser = ArgumentParser(
    description="Produce n-gram mappings or sub_misspells data for the Spellchecking ASR Customization"
)
parser.add_argument(
    "--mode",
    required=True,
    type=str,
    help='Mode, one of ["extract_giza_alignments", "get_replacement_vocab", "get_sub_misspells"]',
)
parser.add_argument("--output_name", required=True, type=str, help='Output file')
parser.add_argument("--input_name", required=True, type=str, help='Input file or folder, depending on mode')
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


def update_vocabs_with_aligned_fragment(
    inputs: List[str],
    replacements: List[str],
    full_vocab: Dict[str, dict],
    src_vocab: Dict[str, int],
    dst_vocab: Dict[str, int],
    clean: bool = False,
) -> None:
    inp = " ".join(inputs)
    rep = " ".join(replacements)
    if clean:
        rep = rep.replace("<DELETE>", "").replace("+", "").replace(" ", "").replace("_", " ")
        inp = inp.replace(" ", "").replace("_", " ")
    if not rep in full_vocab[inp]:
        full_vocab[inp][rep] = 0
    full_vocab[inp][rep] += 1
    src_vocab[inp] += 1
    dst_vocab[rep] += 1


def get_replacement_vocab() -> None:
    """Loops through the file with alignment results, counts frequencies of different replacement segments.
    """

    full_vocab = defaultdict(dict)
    src_vocab = defaultdict(int)
    dst_vocab = defaultdict(int)
    n = 0
    with open(args.input_name, "r", encoding="utf-8") as f:
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
                    update_vocabs_with_aligned_fragment(
                        inputs[begin:end], replacements[begin:end], full_vocab, src_vocab, dst_vocab
                    )

    with open(args.output_name, "w", encoding="utf-8") as out:
        for inp in full_vocab:
            for rep in full_vocab[inp]:
                out.write(
                    inp
                    + "\t"
                    + rep
                    + "\t"
                    + str(full_vocab[inp][rep])
                    + "\t"
                    + str(src_vocab[inp])
                    + "\t"
                    + str(dst_vocab[rep])
                    + "\n"
                )


def get_sub_misspells() -> None:
    """Loops through the file with alignment results, extract aligned segments if they correspond to whole words.
    """
    full_vocab = defaultdict(dict)
    src_vocab = defaultdict(int)
    dst_vocab = defaultdict(int)
    n = 0
    with open(args.input_name, "r", encoding="utf-8") as f:
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
            for i in range(len(inputs)):
                # if corresponding spaces are aligned, this is safe word border
                if inputs[i] == "_" and replacements[i] == "_":
                    update_vocabs_with_aligned_fragment(
                        inputs[begin:i], replacements[begin:i], full_vocab, src_vocab, dst_vocab, clean=True
                    )
                    begin = i + 1
            if begin > 0:  # last fragment until the end
                update_vocabs_with_aligned_fragment(
                    inputs[begin:], replacements[begin:], full_vocab, src_vocab, dst_vocab, clean=True
                )
            # add the whole phrase itself
            update_vocabs_with_aligned_fragment(inputs, replacements, full_vocab, src_vocab, dst_vocab, clean=True)

    with open(args.output_name, "w", encoding="utf-8") as out:
        for inp in full_vocab:
            for rep in full_vocab[inp]:
                if full_vocab[inp][rep] / src_vocab[inp] <= 1 / 200:
                    continue
                if rep == "":
                    continue
                out.write(
                    inp
                    + "\t"
                    + rep
                    + "\t"
                    + str(full_vocab[inp][rep])
                    + "\t"
                    + str(src_vocab[inp])
                    + "\t"
                    + str(dst_vocab[rep])
                    + "\n"
                )


def extract_giza_alignments() -> None:
    # src=reference, dst=misspell
    g = open(args.input_name + "/GIZA++.A3.final", "r", encoding="utf-8")
    f = open(args.input_name + "/GIZA++reverse.A3.final", "r", encoding="utf-8")
    out = open(args.output_name, "w", encoding="utf-8")
    cache = {}
    good_count, not_mono_count, not_covered_count, exception_count = 0, 0, 0, 0
    n = 0
    while True:
        n += 3
        if n % 10000 == 0:
            print(n, "lines processed")
        fline1 = f.readline().strip()
        fline2 = f.readline().strip()
        fline3 = f.readline().strip()
        gline1 = g.readline().strip()
        gline2 = g.readline().strip()
        gline3 = g.readline().strip()
        if fline1 == "" and gline1 == "":
            break
        cache_key = fline1 + "\t" + fline2 + "\t" + gline1 + "\t" + gline2
        if cache_key in cache:
            out.write(cache[cache_key] + "\n")
            continue
        if fline1 == "" or gline1 == "" or fline2 == "" or gline2 == "" or fline3 == "" or gline3 == "":
            raise ValueError("Empty line: " + str(n))
        try:
            matrix, srctokens, dsttokens = fill_alignment_matrix(fline2, fline3, gline2, gline3)
        except Exception:
            print(fline1)
            print(fline2)
            print(fline3)
            print(gline1)
            print(gline2)
            print(gline3)
            exception_count += 1
            out_str = "-exception:\t" + fline2 + "\t" + gline2
            out.write(out_str + "\n")
            continue
        else:
            matrix[matrix <= 2] = 0  # leave only 1-to-1 alignment points
            if check_monotonicity(matrix):
                targets = get_targets(matrix, dsttokens, delimiter="+")
                if len(targets) != len(srctokens):
                    raise ValueError(
                        "targets length doesn't match srctokens length: len(targets)="
                        + str(len(targets))
                        + "; len(srctokens)="
                        + str(len(srctokens))
                    )

                align = " ".join(targets)

                ban = False

                if align is None:
                    ban = True

                if ban:
                    out_str = "ban:\t" + " ".join(srctokens) + "\t" + " ".join(dsttokens) + "\t" + str(align)
                else:
                    out_str = "good:\t" + " ".join(srctokens) + "\t" + " ".join(dsttokens) + "\t" + align

                out.write(out_str + "\n")
                cache[cache_key] = out_str
            else:
                out_str = "-mon:\t" + " ".join(srctokens) + "\t" + " ".join(dsttokens)
                out.write(out_str + "\n")
                cache[cache_key] = out_str
                not_mono_count += 1

    f.close()
    g.close()
    out.close()


def main() -> None:
    if args.mode == "get_replacement_vocab":
        get_replacement_vocab()
    elif args.mode == "get_sub_misspells":
        get_sub_misspells()
    elif args.mode == "extract_alignments":
        extract_giza_alignments()
    else:
        raise ValueError("unknown mode: " + args.mode)


if __name__ == "__main__":
    main()
