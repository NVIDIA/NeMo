# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script can be used to combine joined giza alignments and Google Text Normalization dataset
to produce training corpus for the ThutmoseTaggerModel.
"""

import glob
import os
from argparse import ArgumentParser
from collections import Counter
from nemo.collections.nlp.data.text_normalization_as_tagging.utils import get_src_and_dst_for_alignment
from nemo.utils import logging
from typing import TextIO, Dict, Tuple, Optional

parser = ArgumentParser(description="Produce data for the ThutmoseTaggerModel")
parser.add_argument("--mode", required=True, type=str,
                    help='Mode, one of ["get_replacement_vocab", "filter_by_vocab", "get_labeled_corpus"]')
parser.add_argument("--data_dir", required=True, type=str,
                    help='Path to data directory with files like output-00000-of-00100.tsv')
parser.add_argument("--giza_dir", required=True, type=str,
                    help='Path to directory with class folders like ordinal, date etc')
parser.add_argument("--alignment_filename", required=True, type=str,
                    help='Name of alignment file, like "itn.out", "itn.out.vocab2000"')
parser.add_argument("--out_filename", required=True, type=str, help='Output file')
parser.add_argument("--vocab_filename", required=True, type=str, help='Vocab name')
parser.add_argument("--lang", required=True, type=str, help="Language")
args = parser.parse_args()


def process_file_itn(inputname: str, out: TextIO, keys2replacements: Dict[str, str]) -> None:
    words = []
    tags = []
    semiotic_info = []
    sent_is_ok = True
    with open(inputname, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("<eos>"):
                if sent_is_ok and len(words) > 0:
                    out.write(" ".join(words) + "\t" + " ".join(tags) + "\t" + ";".join(semiotic_info) + "\n")
                words = []
                tags = []
                semiotic_info = []
                sent_is_ok = True
            else:
                cls, written, spoken = line.strip().split("\t")
                if spoken == "sil":
                    continue
                if spoken == "<self>":
                    words.append(written.casefold())
                    tags.append("<SELF>")
                    continue
                src, dst, same_begin, same_end = get_src_and_dst_for_alignment(cls.casefold(), written, spoken,
                                                                               args.lang)
                same_from_begin = [] if same_begin == "" else same_begin.split(" ")
                same_from_end = [] if same_end == "" else same_end.split(" ")
                key = cls.casefold() + "\t" + src + "\t" + dst
                if key in keys2replacements:
                    replacements = keys2replacements[key].split(" ")
                    spoken_words = dst.split(" ")
                    for w, r in zip(same_from_begin + spoken_words + same_from_end,
                                    same_from_begin + replacements + same_from_end):
                        words.append(w)
                        if cls == "LETTERS" or cls == "PLAIN":
                            if w == r:
                                tags.append("<SELF>")
                            else:
                                tags.append(r)
                        elif w == r.replace("_", ""):
                            tags.append("<SELF>")
                        else:
                            tags.append(r)
                    semiotic_info.append(cls + " " + str(len(words)
                                                         - len(spoken_words)
                                                         - len(same_from_begin)
                                                         - len(same_from_end))
                                         + " " + str(len(words)))
                else:
                    sent_is_ok = False


def process_file_tn(inputname: str, out: TextIO, keys2replacements: Dict[str, str]) -> None:
    words = []
    tags = []
    semiotic_info = []
    sent_is_ok = True
    with open(inputname, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("<eos>"):
                if sent_is_ok and len(words) > 0:
                    out.write(" ".join(words) + "\t" + " ".join(tags) + "\t" + ";".join(semiotic_info) + "\n")
                words = []
                tags = []
                semiotic_info = []
                sent_is_ok = True
            else:
                cls, written, spoken = line.strip().split("\t")
                if spoken == "sil":
                    if " " in written:     # this means there is an error in corpus, will lead to token number mismatch
                        sent_is_ok = False
                    else:
                        words.append(written.casefold())
                        tags.append("<DELETE>")
                    continue
                if spoken == "<self>":
                    words.append(written.casefold())
                    tags.append("<SELF>")
                    continue
                src, dst, same_begin, same_end = get_src_and_dst_for_alignment(cls.casefold(),
                                                                               written, spoken, args.lang)
                same_from_begin = [] if same_begin == "" else same_begin.split(" ")
                same_from_end = [] if same_end == "" else same_end.split(" ")
                key = cls.casefold() + "\t" + src + "\t" + dst
                if key in keys2replacements:
                    replacements = keys2replacements[key].split(" ")
                    spoken_words = dst.split(" ")
                    for w, r in zip(same_from_begin + replacements + same_from_end,
                                    same_from_begin + spoken_words + same_from_end):
                        words.append(w)
                        if cls == "LETTERS" or cls == "PLAIN":
                            if w == r:
                                tags.append("<SELF>")
                            else:
                                tags.append(r)
                        elif w == r.replace("_", ""):
                            tags.append("<SELF>")
                        else:
                            tags.append(r)
                    semiotic_info.append(cls + " " + str(len(words)
                                                         - len(replacements)
                                                         - len(same_from_begin)
                                                         - len(same_from_end))
                                         + " " + str(len(words)))
                else:
                    sent_is_ok = False


def process_line(semiotic_class: str, line: str) -> Optional[Tuple[str, str, str, int]]:
    parts = line.strip().split("\t")
    if len(parts) != 6:
        return None
    freq = int(parts[0])
    if parts[1] != "good:":
        return None

    src, dst, leftside_align, rightside_align = parts[2], parts[3], parts[4], parts[5]
    align = rightside_align
    if semiotic_class == "letters" or semiotic_class == "plain":
        align = leftside_align

    return src, dst, align, freq


def get_replacement_vocab(tn: bool = False) -> None:
    full_vocab = Counter()
    alignment_files = glob.glob(args.giza_dir + "/*/" + args.alignment_filename)
    for fn in alignment_files:
        fn_parts = fn.split("/")
        assert (len(fn_parts) >= 2)
        semiotic_class = fn_parts[-2]
        class_vocab = Counter()
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                t = process_line(semiotic_class, line)
                if t is None:
                    continue
                src, dst, replacement, freq = t
                inputs = src.split(" ")
                replacements = replacement.split(" ")
                assert (len(inputs) == len(replacements)), "mismatch in: " + line
                for inp, rep in zip(inputs, replacements):
                    if inp == rep:  # skip same words
                        continue
                    if tn:
                        full_vocab[inp] += freq
                    else:  # itn
                        full_vocab[rep] += freq
                        class_vocab[rep] += freq
        with open(args.vocab_filename + "." + semiotic_class, "w", encoding="utf-8") as out:
            for k, v in class_vocab.most_common(1000000000):
                out.write(k + "\t" + str(v) + "\n")

    with open(args.vocab_filename, "w", encoding="utf-8") as out:
        for k, v in full_vocab.most_common(1000000000):
            out.write(k + "\t" + str(v) + "\n")


def filter_by_vocab(tn: bool = False) -> None:
    if not os.path.exists(args.vocab_filename):
        raise ValueError(f"Alignments dir {args.giza_dir} does not exist")
    # load vocab from file
    vocab = {}
    with open(args.vocab_filename, "r", encoding="utf-8") as f:
        for line in f:
            k, v = line.strip().split("\t")
            vocab[k] = int(v)
    print("len(vocab)=", len(vocab))
    alignment_files = glob.glob(args.giza_dir + "/*/" + args.alignment_filename)
    for fn in alignment_files:
        fn_parts = fn.split("/")
        assert (len(fn_parts) >= 2)
        semiotic_class = fn_parts[-2]
        out = open(args.giza_dir + "/" + semiotic_class + "/" + args.out_filename, "w", encoding="utf8")
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                t = process_line(semiotic_class, line)
                if t is None:
                    continue
                src, dst, replacement, freq = t
                ok = True
                for s, r in zip(src.split(" "), replacement.split(" ")):
                    if s != r:
                        if tn and s not in vocab:
                            ok = False
                        elif not tn and r not in vocab:
                            ok = False
                if ok:
                    out.write(semiotic_class + "\t" + src + "\t" + dst + "\t" + replacement + "\n")
        out.close()


def get_labeled_corpus(tn: bool = False) -> None:
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data dir {args.data_dir} does not exist")

    keys2replacements = {}
    alignment_files = glob.glob(args.giza_dir + "/*/" + args.alignment_filename)
    for af in alignment_files:
        with open(af, "r", encoding="utf-8") as f:
            for line in f:
                cls, src, dst, replacements = line.strip().split("\t")
                key = cls + "\t" + dst + "\t" + src
                if key in keys2replacements and keys2replacements[key] != replacements:
                    logging.warning("keys2replacements[key] != replacements", keys2replacements[key], replacements)
                keys2replacements[key] = replacements
    print(len(keys2replacements))
    out = open(args.out_filename, "w", encoding="utf-8")
    input_paths = sorted([os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)])
    for inputname in input_paths:
        if tn:
            process_file_tn(inputname, out, keys2replacements)
        else:
            process_file_itn(inputname, out, keys2replacements)
    out.close()


def main() -> None:
    if not os.path.exists(args.giza_dir):
        raise ValueError(f"Alignments dir {args.giza_dir} does not exist")

    if args.mode == "get_replacement_vocab":
        get_replacement_vocab()
    elif args.mode == "get_replacement_vocab_tn":
        get_replacement_vocab(tn=True)
    elif args.mode == "filter_by_vocab":
        filter_by_vocab()
    elif args.mode == "filter_by_vocab_tn":
        filter_by_vocab(tn=True)
    elif args.mode == "get_labeled_corpus":
        get_labeled_corpus()
    elif args.mode == "get_labeled_corpus_tn":
        get_labeled_corpus(tn=True)
    else:
        assert(False), "unknown mode"


if __name__ == "__main__":
    main()
