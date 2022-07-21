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
This script can be used to combine joined giza alignments and Google Text Normalization dataset
to produce training corpus for the ThutmoseTaggerModel.
"""

import glob
import os
import re
from argparse import ArgumentParser
from collections import Counter
from typing import Dict, Optional, TextIO, Tuple

from nemo.collections.nlp.data.text_normalization_as_tagging.utils import get_src_and_dst_for_alignment
from nemo.collections.nlp.data.text_normalization_as_tagging.utils import alpha_tokenize
from nemo.utils import logging

parser = ArgumentParser(description="Produce data for the ThutmoseTaggerModel")
parser.add_argument(
    "--mode",
    required=True,
    type=str,
    help='Mode, one of ["get_replacement_vocab", "filter_by_vocab", "get_labeled_corpus"]',
)
parser.add_argument(
    "--data_dir", required=True, type=str, help='Path to data directory with files like output-00000-of-00100.tsv'
)
parser.add_argument(
    "--giza_dir", required=True, type=str, help='Path to directory with class folders like ordinal, date etc'
)
parser.add_argument(
    "--alignment_filename", required=True, type=str, help='Name of alignment file, like "itn.out", "itn.out.vocab2000"'
)
parser.add_argument("--out_filename", required=True, type=str, help='Output file')
parser.add_argument("--vocab_filename", required=True, type=str, help='Vocab name')
parser.add_argument("--lang", required=True, type=str, help="Language")
args = parser.parse_args()


def process_file(inputname: str, out: TextIO, keys2replacements: Dict[str, str], tn: bool = False) -> None:
    """Processes one file in Google TN Dataset format to get the labeled data for ThutmoseTaggerModel

    Args:
        inputname: name of input file
        out: output stream
        keys2replacements: Mapping from (semiotic class, spoken, written) to the segmented written form,
         which is aligned one-to-one to spoken words (this is the result obtained from Giza++ alignment pipeline)

    """
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
                    if tn:
                        if " " in written:  # this means there is an error in corpus, will lead to token number mismatch
                            sent_is_ok = False
                        else:
                            words.append(written.casefold())
                            tags.append("<DELETE>")
                    continue
                if spoken == "<self>":
                    words.append(written.casefold())
                    tags.append("<SELF>")
                    continue
                # In TN leave abbreviations as is: e.g. "LETTERS     ATM     a t m"
                if tn and cls == "LETTERS" and written.isalpha():
                    words.append(written.casefold())
                    tags.append("<SELF>")
                    continue
                # Leave plain orphovariants as is: e.g. PLAIN   popularised     popularized
                if tn and cls == "PLAIN" and written.isalpha():
                    words.append(written.casefold())
                    tags.append("<SELF>")
                    continue
                src, dst, same_begin, same_end = get_src_and_dst_for_alignment(
                    cls.casefold(), written, spoken, args.lang
                )
                same_from_begin = [] if same_begin == "" else same_begin.split(" ")
                same_from_end = [] if same_end == "" else same_end.split(" ")
                key = cls.casefold() + "\t" + src + "\t" + dst
                if key in keys2replacements:
                    replacements_str = keys2replacements[key]
                    replacements = replacements_str.split(" ")
                    spoken_words = dst.split(" ")
                    inputs, targets = spoken_words, replacements
                    if tn:
                        inputs, targets = replacements, spoken_words
                    for w, r in zip(
                        same_from_begin + inputs + same_from_end,
                        same_from_begin + targets + same_from_end
                    ):
                        words.append(w)
                        if cls == "LETTERS" or cls == "PLAIN":
                            if w == r:
                                tags.append("<SELF>")
                            else:
                                tags.append(r)
                        if tn and cls == "TELEPHONE":  # correct google corpus issue
                            if r == "sil":
                                tags.append("<DELETE>")
                        elif w == r.replace("_", ""):
                            tags.append("<SELF>")
                        else:
                            tags.append(r)
                    semiotic_info.append(
                        cls
                        + " "
                        + str(len(words) - len(inputs) - len(same_from_begin) - len(same_from_end))
                        + " "
                        + str(len(words))
                    )
                    if len(words) != len(tags):
                        print(
                            "WARNING: len(words)=" + str(len(words)) + "; len(tags)=" + str(len(tags)) + "; line=" + line
                        )
                        sent_is_ok = False
                else:
                    sent_is_ok = False


def process_file_tn_tokenizer(inputname: str, out: TextIO, keys2replacements: Dict[str, str]) -> None:
    """Processes one file in Google TN Dataset format to get the labeled data for ThutmoseTaggerTNTokenizerModel

    Args:
        inputname: name of input file
        out: output stream
        keys2replacements: Mapping from (semiotic class, spoken, written) to the
        (written-tokenized-by-alpha-num, written-tokenizer-tags)

    """
    words = ["<BOS>"]
    labels = ["SPACE"]
    semiotic_info = []
    sent_is_ok = True
    with open(inputname, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("<eos>"):
                if sent_is_ok and len(words) > 0:
                    out.write(" ".join(words) + "\t" + " ".join(labels) + "\t" + ";".join(semiotic_info) + "\n")
                words = ["<BOS>"]
                labels = ["SPACE"]
                semiotic_info = []
                sent_is_ok = True
            else:
                cls, written, spoken = line.strip().split("\t")
                if spoken == "sil":
                    continue
                if spoken == "<self>":
                    words.append(written.casefold())
                    labels.append("SPACE")
                    continue
                # Leave abbreviations as is: e.g. "LETTERS     ATM     a t m"
                if cls == "LETTERS" and written.isalpha():
                    words.append(written.casefold())
                    labels.append("SPACE")
                    continue
                # Leave plain orphovariants as is e.g. PLAIN   popularised     popularized
                if cls == "PLAIN" and written.isalpha():
                    words.append(written.casefold())
                    labels.append("SPACE")
                    continue
                src, dst, same_begin, same_end = get_src_and_dst_for_alignment(
                    cls.casefold(), written, spoken, args.lang
                )
                same_from_begin = [] if same_begin == "" else same_begin.split(" ")
                same_from_end = [] if same_end == "" else same_end.split(" ")
                key = cls.casefold() + "\t" + src + "\t" + dst
                if key in keys2replacements:
                    written_tokens_str, token_labels_str = keys2replacements[key]
                    written_tokens = written_tokens_str.split(" ")
                    token_labels = token_labels_str.split(" ")
                    if len(token_labels) != len(written_tokens) + 1:
                        raise ValueError(
                            "token-labels mismatch: len(token_labels)="
                            + str(len(token_labels))
                            + "; len(written_tokens)="
                            + str(len(written_tokens))
                            + "; tokens="
                            + written_tokens_str
                            + "; labels="
                            + token_labels_str
                            + "; key="
                            + key
                        )
                    for w in same_from_begin:
                        words.append(w)
                        labels.append("SPACE")

                    # handle +1 labels issue
                    labels.pop()

                    words.extend(written_tokens)
                    labels.extend(token_labels)

                    for w in same_from_end:
                        words.append(w)
                        labels.append("SPACE")

                    semiotic_info.append(
                        cls
                        + " "
                        + str(len(words) - len(written_tokens) - len(same_from_begin) - len(same_from_end))
                        + " "
                        + str(len(words))
                    )
                else:
                    sent_is_ok = False


def process_line(semiotic_class: str, line: str) -> Optional[Tuple[str, str, str, int]]:
    """A helper function to read the file with alignment results"""

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
    """Loops through the files with alignment results in each semiotic class subfolder, counts frequencies of different
     replacement segments.
    """

    full_vocab = Counter()
    alignment_files = glob.glob(args.giza_dir + "/*/" + args.alignment_filename)
    for fn in alignment_files:
        fn_parts = fn.split("/")
        if len(fn_parts) < 2:
            raise ValueError("Bad filename: " + fn)
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
                if len(inputs) != len(replacements):
                    raise ValueError("Length mismatch in: " + line)
                for inp, rep in zip(inputs, replacements):
                    if inp == rep:  # skip same words
                        continue
                    if tn:
                        full_vocab[inp] += freq
                        class_vocab[inp] += freq

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
    """Given a restricted vocabulary of replacements,
    loops through the files with alignment results in each semiotic class subfolder,
    discards the examples containing a replacement which is not in our restricted vocabulary.
    """

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
        if len(fn_parts) < 2:
            raise ValueError("Bad filename: " + fn)
        semiotic_class = fn_parts[-2]
        out = open(args.giza_dir + "/" + semiotic_class + "/" + args.out_filename, "w", encoding="utf-8")
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
    """Loops through the files with alignment results in each semiotic class subfolder,
    collects a mapping from (semiotic class, spoken, written) to the segmented written form,
         which is aligned one-to-one to spoken words.
    Then loops through the files in Google TN Dataset format to get the labeled data for ThutmoseTaggerModel.
    It extracts the whole sentences and substitutes the semiotic spans to their aligned form from the dictionary.
    """

    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data dir {args.data_dir} does not exist")

    keys2replacements = {}
    alignment_files = glob.glob(args.giza_dir + "/*/" + args.alignment_filename)
    if len(alignment_files) == 0:
        raise ValueError("Did not found any such files: " + args.giza_dir + "/*/" + args.alignment_filename)
    for af in alignment_files:
        if tn and "electronic" in af:
            continue
        with open(af, "r", encoding="utf-8") as f:
            for line in f:
                cls, src, dst, replacements = line.strip().split("\t")
                if tn:
                    replacements = replacements.replace("_", "").replace("<<", "")  # remove special tokenization symbols
                    if replacements == "" or replacements == " ":
                        print("WARNING: empty replacements: ", line)
                        continue
                key = cls + "\t" + dst + "\t" + src
                if key in keys2replacements and keys2replacements[key] != replacements:
                    logging.warning("keys2replacements[key] != replacements", keys2replacements[key], replacements)
                keys2replacements[key] = replacements
    print("size of phrase-to-replacements dictionary =", len(keys2replacements))
    out = open(args.out_filename, "w", encoding="utf-8")
    input_paths = sorted([os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)])
    for inputname in input_paths:
        if tn:
            process_file(inputname, out, keys2replacements, tn=True)
        else:
            process_file(inputname, out, keys2replacements, tn=False)
    out.close()


def get_labeled_corpus_for_tn_tokenizer() -> None:
    """Loops through the files with alignment results in each semiotic class subfolder,
    collects a mapping from (semiotic class, spoken, written) to the (written-tokenized-by-alpha-num, written-tokenizer-tags).
    Skip PLAIN and LETTERS semiotic classes.
    Then loops through the files in Google TN Dataset format to get the whole labeled sentences.
    It extracts the whole sentences and substitutes the semiotic spans to their tokenization tags from the dictionary.

      JOIN - next token should come without space
      SPACE - next token should be separated by space
      DUMMY1 - empty node <DELETE> should be inserted before the next token
      DUMMY2 - two empty nodes <DELETE> should be inserted before the next token
      DUMMY3 - three empty nodes <DELETE> should be inserted before the next token
      DUMMY4 - four empty nodes <DELETE> should be inserted before the next token
    """

    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data dir {args.data_dir} does not exist")
    keys2replacements = {}
    alignment_files = glob.glob(args.giza_dir + "/*/" + args.alignment_filename)
    if len(alignment_files) == 0:
        raise ValueError("Did not found any such files: " + args.giza_dir + "/*/" + args.alignment_filename)
    for af in alignment_files:
        with open(af, "r", encoding="utf-8") as f:
            for line in f:
                # money     twenty four thousand six hundred fifteen dollars    _2 4 , 6 1 5_ _$<<    _2 4 , 6 <DELETE> 15_ _$<<
                cls, spoken_words_str, written_tokens_str, written_segments_str = line.strip().split("\t")
                key = cls + "\t" + written_tokens_str + "\t" + spoken_words_str

                # skip lines with 5 <DELETE> or more
                if "<DELETE> <DELETE> <DELETE> <DELETE> <DELETE>" in written_segments_str:
                    continue
                written_segments_str = re.sub(r"<DELETE> <DELETE> <DELETE> <DELETE>", r"DUMMY4", written_segments_str)
                written_segments_str = re.sub(r"<DELETE> <DELETE> <DELETE>", r"DUMMY3", written_segments_str)
                written_segments_str = re.sub(r"<DELETE> <DELETE>", r"DUMMY2", written_segments_str)
                written_segments_str = re.sub(r"<DELETE>", r"DUMMY1", written_segments_str)

                written_segments_str = written_segments_str.replace("_", "").replace("<", "").replace(">", "")
                written_tokens_str = written_tokens_str.replace("_", "").replace("<", "").replace(">", "")

                if written_segments_str == "" or written_tokens_str == "":
                    continue

                # need to re-tokenize some preprocessed cases, like "3th", "oct.", so can't just split by space
                alpha_tokens = alpha_tokenize(written_tokens_str)

                written_segments = written_segments_str.split(" ")  # 2 4 , 6 <DELETE> 15 $
                token_labels = []
                current = ""
                ok = True

                i = 0
                j = 0
                while len(token_labels) < len(alpha_tokens):
                    if i >= len(written_segments):
                        raise IndexError(
                            "i=" + str(i) + "; written_segments=" + str(written_segments) + "; token_labels=" + str(
                                token_labels) + "; line=" + line)
                    segment = written_segments[i]
                    i += 1

                    if segment.startswith("DUMMY"):
                        token_labels.append(segment)
                        segment = written_segments[i]
                        i += 1
                    else:
                        token_labels.append("SPACE")

                    while (
                        current != segment
                        # and not (current == "1/2" and segment == "½")
                        and j < len(alpha_tokens)
                    ):
                        if current != "":
                            token_labels.append("JOIN")
                        current += alpha_tokens[j]
                        j += 1
                        if len(current) > len(segment):
                            logging.warning("mismatch: segment=", segment, "current=", current, " line=" + line)
                            ok = False
                            break
                    if current != segment:
                        logging.warning("mismatch: segment=", segment, "current=", current, " line=" + line)
                        ok = False
                        break
                    current = ""

                token_labels.append("SPACE")  # there are +1 labels compared to tokens because of beginning and end

                if not ok:
                    continue

                if len(token_labels) != len(alpha_tokens) + 1:
                    raise IndexError("len mismatch: " + line + "; labels=" + str(token_labels))

                value = (" ".join(alpha_tokens), " ".join(token_labels))
                if key in keys2replacements and keys2replacements[key] != value:
                    logging.warning(
                        "keys2replacements[key] != replacements",
                        keys2replacements[key],
                        written_tokens_str,
                        " ".join(token_labels)
                    )
                keys2replacements[key] = value
    print("size of phrase-to-replacements dictionary =", len(keys2replacements))
    out = open(args.out_filename, "w", encoding="utf-8")
    input_paths = sorted([os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)])
    for inputname in input_paths:
        process_file_tn_tokenizer(inputname, out, keys2replacements)
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
    elif args.mode == "get_labeled_corpus_for_tn_tokenizer":
        get_labeled_corpus_for_tn_tokenizer()
    else:
        raise ValueError("unknown mode: " + args.mode)


if __name__ == "__main__":
    main()
