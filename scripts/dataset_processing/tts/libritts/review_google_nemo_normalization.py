# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import re
import warnings
import argparse

from collections import Counter
from pathlib import Path

import json
from tqdm import tqdm

# this module is copy-paste from https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch/common/text/unidecoder
from scripts.dataset_processing.tts.libritts.unidecoder import unidecoder

parser = argparse.ArgumentParser(description='TODO')
parser.add_argument("--google-normalized-manifest-path", required=True, type=str)
parser.add_argument("--nemo-normalized-manifest-path", required=True, type=Path)
parser.add_argument("--save-dir", type=Path)

args = parser.parse_args()

# Normalized text in LibriTTS by Google which contains abbreviations from `libri_only_remove_dot_abbrs` looks like this:
# "Mr. Smith" -> "mr Smith" (i.e removing dot and lowercasing all letters).
libri_only_remove_dot_abbrs = {
    "Mrs.", "Mr.", "Dr.", "Co.", "Lt.",
    "Sgt.", "Drs.", "Maj.", "Capt.", "Esq.",
    "Gen.", "Ltd.", "Col."
}

# Normalized text in LibriTTS by Google which contains abbreviations from `libri_converts_abbrs` looks like this:
# "&" -> "and", "Jr." -> "Junior" (i.e correct conversion).
libri_converts_abbrs = {
    "&", "Gov.", "=", "Jr.", "Hon.", "Mt.",
    "§"  # currently, unidecoder doesn't pass it
}

# Normalized text in LibriTTS by Google which contains abbreviations from `libri_sometimes_converts_abbrs` sometimes wasn't converted.
libri_sometimes_converts_abbrs = {"St.": "saint", "Rev.": "reverend"}

# Normalized text in LibriTTS by Google which contains abbreviations from `libri_wo_changes_abbrs` wasn't converted.
libri_wo_changes_abbrs = {"vs.": "versus"}


google_abbr2expand = {
    "mr": "mister",
    "Mr": "Mister",
    "mrs": "misses",
    "Mrs": "Misses",
    "dr": "doctor",
    "Dr": "Doctor",
    "drs": "doctors",
    "Drs": "Doctors",
    "co": "company",
    "Co": "Company",
    "lt": "lieutenant",
    "Lt": "Lieutenant",
    "sgt": "sergeant",
    "Sgt": "Sergeant",
    "st": "saint",
    "St": "Saint",
    "jr": "junior",
    "Jr": "Junior",
    "maj": "major",
    "Maj": "Major",
    "hon": "honorable",
    "Hon": "Honorable",
    "gov": "governor",
    "Gov": "Governor",
    "capt": "captain",
    "Capt": "Captain",
    "esq": "esquire",
    "Esq": "Esquire",
    "gen": "general",
    "Gen": "General",
    "ltd": "limited",
    "Ltd": "Limited",
    "rev": "reverend",
    "Rev": "Reverend",
    "col": "colonel",
    "Col": "Colonel",
    "mt": "mount",
    "Mt": "Mount",
    "ft": "fort",
    "Ft": "Fort",
    "tenn": "tennessee",
    "Tenn": "Tennessee",
    "vs": "versus",
    "Vs": "Versus",
    "&": "and",
    "§": "section",
    "#": "hash",
    "=": "equals"
}

nemo_abbr2expand = {
    "Mr.": "mister",
    "Mrs.": "misses",
    "Dr.": "doctor",
    "Co.": "company",
    "Lt.": "lieutenant",
    "Sgt.": "sergeant",
    "St.": "saint",
    "Jr.": "junior",
    "Maj.": "major",
    "Hon.": "honorable",
    "Gov.": "governor",
    "Capt.": "captain",
    "Esq.": "esquire",
    "Gen.": "general",
    "Ltd.": "limited",
    "Rev.": "reverend",
    "Col.": "colonel",
    "Mt.": "mount",
    "Ft.": "fort",
    "Tenn.": "tennessee",
    "vs.": "versus",
    "&": "and",
    "§": "section",
    "#": "hash",
    "=": "equals"
}


def counter_to_key(c):
    c2num = sorted(c.items(), key=lambda n: n[0], reverse=False)
    return "|".join([f"{num} x `{c}`" for (c, num) in c2num if num != 0])


def almost_equal(a, b, stats=None):
    # check if `a` and `b` are almost equal (except for dots, commas, hyphens and spaces).

    original_a, original_b = a, b

    marks_a = re.findall(r'[\s\,\.\-]', a)
    marks_a = "".join(sorted(marks_a))

    a = re.sub('[\s\,\.\-]', '', a)

    marks_b = re.findall(r'[\s\,\.\-]', b)
    marks_b = "".join(sorted(marks_b))

    b = re.sub('[\s\,\.\-]', '', b)

    cntr_a = Counter(a)
    cntr_b = Counter(b)

    if cntr_a == cntr_b:
        if stats is not None:
            cntr_marks_a = Counter(marks_a)
            cntr_marks_b = Counter(marks_b)
            cntr_marks_a.subtract(cntr_marks_b)
            key = f"{counter_to_key(cntr_marks_a)}"
            if key not in stats:
                stats[key] = 0
            stats[key] += 1

            # if key == '1 x `.`':
            #     print()
            #     print(f"|{original_a}|")
            #     print(f"|{original_b}|")

        return True

    return False


def bug_with_one_first(a, b):
    a_words = Counter(re.findall(r'\w+', a))
    b_words = Counter(re.findall(r'\w+', b))

    b_words.subtract(a_words)
    b_words = Counter({k: v for k, v in b_words.items() if v != 0})

    if set(b_words.keys()) <= {"i", "first", "one"} and b_words["i"] != 0:
        if b_words["i"] + b_words["first"] + b_words["one"] == 0:
            return True

    return False


def save(data_list, path):
    with open(path, "w") as f:
        f.write("\n".join(["|".join(str(i) for i in data) for data in data_list]))


def main():
    stats = {}

    good_samples = []
    bad_characters_samples = []
    different_samples = []
    almost_equal_samples = []
    bug_with_one_first_samples = []

    google_texts = []
    with open(args.google_normalized_manifest_path) as f:
        for l in f:
            j = json.loads(l)
            text = j["normalized_text"].strip()
            key = j["audio_filepath"]
            google_texts.append((key, text))

    google_texts = sorted(google_texts, key=lambda x: x[0])

    nemo_texts = []
    with open(args.nemo_normalized_manifest_path) as f:
        for l in f:
            j = json.loads(l)
            text = j["normalized_text"].strip()
            key = j["audio_filepath"]
            nemo_texts.append((key, text))

    nemo_texts = sorted(nemo_texts, key=lambda x: x[0])

    assert len(google_texts) == len(nemo_texts)

    for i, (g, n) in tqdm(enumerate(zip(google_texts, nemo_texts))):
        audio_path, google_text = g[0], g[1]
        nemo_text = n[1]

        # let's skip samples with bad characters
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                google_text = unidecoder(google_text)
                nemo_text = unidecoder(nemo_text)
            except UserWarning as _:
                bad_characters_samples.append((i, nemo_text, google_text))
                continue

        ###########################################
        # Additional text pre-processing for nemo #
        ###########################################

        # let's fix `librivox.org` manually, because nemo normalizer sometimes doesn't normalize or splits librivox by symbol
        nemo_text = nemo_text.replace("librivox.org", "librivox dot org")
        nemo_text = nemo_text.replace("l i b r i v o x dot org", "librivox dot org")

        # let's fix `one/first'am` and `one/first'll` manually, because nemo normalizer has problem with it
        nemo_text = nemo_text.replace("one'll", "I'll")
        nemo_text = nemo_text.replace("first'll", "I'll")
        nemo_text = nemo_text.replace("one'm", "I'm")
        nemo_text = nemo_text.replace("first'm", "I'm")

        # let's normalize `nemo_abbr2expand` abbreviations again, because nemo sometimes forgets to expand them
        for abbr in nemo_abbr2expand.keys():
            if abbr in nemo_text:
                # replace abbr in nemo text via regex and using \b to match only whole words, keep original 1 and 2 groups
                nemo_text = re.sub(rf'(^|\b|\W){abbr}($|\b|\s)', rf"\1{nemo_abbr2expand[abbr]}\2", nemo_text)

        #############################################
        # Additional text pre-processing for google #
        #############################################

        # let's normalize `libri_only_remove_dot_abbrs` abbreviations, because google doesn't do it well
        for abbr in google_abbr2expand.keys():
            if abbr in google_text:
                # replace abbr in google text via regex and using \b to match only whole words, keep original 1 and 2 groups
                google_text = re.sub(rf'(^|\s|\W){abbr}($|\s)', rf"\1{google_abbr2expand[abbr]}\2", google_text)

        # let's normalize `libri_sometimes_converts_abbrs` abbreviations manually, google sometimes forgets to expand them
        for abbr, t in libri_sometimes_converts_abbrs.items():
            google_text = google_text.replace(abbr, t)

        # let's normalize `libri_wo_changes_abbrs` abbreviations manually, google doesn't change, but they should be
        for abbr, t in libri_wo_changes_abbrs.items():
            google_text = google_text.replace(abbr, t)

        #############################################
        # comparison nemo and google normalization  #
        #############################################

        if nemo_text.lower() != google_text.lower():
            # check if nemo_text is almost equal to google_text (with some exceptions, see `almost_equal`)
            if almost_equal(nemo_text.lower(), google_text.lower(), stats):
                almost_equal_samples.append((i, audio_path, nemo_text, google_text))
            # check if it is bug only with I -> first/one in nemo_text
            elif bug_with_one_first(nemo_text.lower(), google_text.lower()):
                bug_with_one_first_samples.append((i, audio_path, nemo_text, google_text))
            else:
                different_samples.append((i, audio_path, nemo_text, google_text))
        else:
            good_samples.append((i, audio_path, nemo_text, google_text))

    # for _, k, n, g in different_samples:
    #     print(k)
    #     print(n)
    #     print(g)
    #     print()
    print()
    print("good samples", len(good_samples), f"{len(good_samples) / len(google_texts) * 100:.2f}%")
    print("almost equal samples (they are good samples too)", len(almost_equal_samples), f"{len(almost_equal_samples) / len(google_texts) * 100:.2f}%")
    print("bad characters samples", len(bad_characters_samples), f"{len(bad_characters_samples) / len(google_texts) * 100:.2f}%")
    print("different_samples", len(different_samples), f"{len(different_samples) / len(google_texts) * 100:.2f}%")
    print("bug_with_one_first_samples", len(bug_with_one_first_samples), f"{len(bug_with_one_first_samples) / len(google_texts) * 100:.2f}%")
    print()

    top_k = 5
    print(f"stats for almost equal samples (top-{top_k}):")
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    top_stats = sorted_stats[:top_k]
    for k, v in top_stats:
        print(f"diff:[{k}], {v} samples, {v / len(google_texts) * 100:.2f}%")

    print(f"stats for almost equal samples (after top-{top_k}):")
    other_stats = sorted_stats[top_k:]
    num_other_samples = sum([v for k, v in other_stats])
    print(f"{num_other_samples} samples, {num_other_samples / len(google_texts) * 100:.2f}%")

    # with open(args.save_dir / "tags_of_good_samples.txt", "w") as f:
    #     f.write("\n".join(tags_of_good_samples))

    assert args.save_dir is not None
    args.save_dir.mkdir(exist_ok=True, parents=True)
    save(bad_characters_samples, args.save_dir / "bad_characters_samples.txt")
    save(different_samples, args.save_dir / "different_samples.txt")
    save(almost_equal_samples, args.save_dir / "almost_equal_samples.txt")


if __name__ == '__main__':
    main()


# General bugs:
# pm/am-like bugs (we can fix it via NeMo rules)

# General bugs in dataset:
# dash is double hyphen, and sometimes it doesn't have space around it

# Google bugs (not covered in code):
# -- -> -
# - -> <empty>
# B.A. -> b a (and it will be hard to fix!)

# NeMo bugs (not covered in code):
# MR. -> mister
# Ky. -> kentucky.
# Mo. -> missouri.
# IND. -> indiana
# Ga. -> georgia

# Example of run:
# export SET="train_other_500"
# python scripts/dataset_processing/tts/libritts/review_google_nemo_normalization.py --google-normalized-manifest-path /data_4tb/datasets/LibriTTS/${SET}_google.json --nemo-normalized-manifest-path /data_4tb/datasets/LibriTTS/${SET}_stt_en_citrinet_1024_normalized.json --save-dir /data_4tb/datasets/LibriTTS/${SET}_review_data