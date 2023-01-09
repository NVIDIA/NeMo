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

from difflib import SequenceMatcher
import json
import os
import pdb

from argparse import ArgumentParser
from collections import Counter
from tqdm.auto import tqdm
from nemo.collections.asr.parts.utils.manifest_utils import (
    read_manifest,
    write_manifest,
)

parser = ArgumentParser(
    description="Analyze custom phrases recognition after ASR"
)
parser.add_argument("--manifest", required=True, type=str, help="Path to manifest file with reference text and asr hypotheses")
parser.add_argument("--vocab_dir", type=str, required=True, help="Path to directory with custom vocabularies")
parser.add_argument("--output_name", type=str, required=True, help="Output file")

args = parser.parse_args()


def get_changed_fragments(i_words, j_words):
    s = SequenceMatcher(None, i_words, j_words)
    result = []
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag != "equal":
            result.append((tag, " ".join(i_words[i1:i2]), " ".join(j_words[j1:j2]), i1, i2, j1, j2))
    result = sorted(result, key=lambda x: x[3])
    return result


def read_custom_vocab(filename):
    phrases = set()
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            phrases.add(line.strip().casefold())
    phrase2id = {}
    id2phrase = {}
    lastid = 0
    for phrase in sorted(phrases, key=lambda x: len(x), reverse=True):
        if phrase not in phrase2id:
            newid = "phraseid" + str(lastid)
            phrase2id[phrase] = newid
            id2phrase[newid] = phrase
            lastid += 1
    return phrases, phrase2id, id2phrase


custom_vocabs = {}  # key=doc_id, value=tuple(set of phrases, phrase2id, id2phrase)

for name in os.listdir(args.vocab_dir):
    parts = name.split(".")
    if len(parts) == 3 and parts[1] == "custom":
        doc_id = parts[0]
        custom_vocabs[doc_id] = read_custom_vocab(os.path.join(args.vocab_dir, name))

test_data = read_manifest(args.manifest)

# extract just the text corpus from the manifest
pred_text = [data['pred_text'] for data in test_data]
ref_text = [data['text'] for data in test_data]
audio_filepath = [data['audio_filepath'] for data in test_data]

out_target_lost = open(args.output_name + ".target_lost", "w", encoding="utf-8")
out_target_worse = open(args.output_name + ".target_worse", "w", encoding="utf-8")
out_target_better = open(args.output_name + ".target_better", "w", encoding="utf-8")
out_target_good = open(args.output_name + ".target_good", "w", encoding="utf-8")
out_target_fp = open(args.output_name + ".target_fp", "w", encoding="utf-8")
out_ideal = open(args.output_name + ".ideal", "w", encoding="utf-8")

count_lost = 0
count_worse = 0
count_better = 0
count_good = 0
count_fp = 0

with open(args.output_name, "w", encoding="utf-8") as out:
    diff_vocab_after_spellcheck_vs_ref = Counter()
    diff_vocab_before_spellcheck_vs_ref = Counter()
    diff_vocab_before_after_spellcheck = Counter()

    for i in range(len(test_data)):
        path = audio_filepath[i]
        # example of path: ...clips/197_0000.wav   #doc_id=197
        path_parts = path.split("/")
        path_parts2 = path_parts[-1].split("_")
        doc_id = path_parts2[-2]
        vocab, phrase2id, id2phrase = custom_vocabs[doc_id]
        ref_sent = " " + ref_text[i] + " "
        after_sent = " " + pred_text[i] + " "
        before_sent = after_sent
        if "before_spell_pred" in test_data[i]:
            before_sent = " " + test_data[i]["before_spell_pred"] + " "

        for phrase in sorted(vocab, key=lambda x: len(x), reverse=True):
            phrase_id = phrase2id[phrase]
            ref_sent = ref_sent.replace(" " + phrase + " ", " " + phrase_id + " ")
            before_sent = before_sent.replace(" " + phrase + " ", " " + phrase_id + " ")
            after_sent = after_sent.replace(" " + phrase + " ", " " + phrase_id + " ")

        before_words = before_sent.strip().split()
        after_words = after_sent.strip().split()
        ref_words = ref_sent.strip().split()

        ideal_words = before_words[:]  # words after ideal spellchecker

        for tag, hyp_fragment, ref_fragment, i1, i2, j1, j2 in get_changed_fragments(after_words, ref_words):
            if hyp_fragment in id2phrase:
                hyp_fragment = id2phrase[hyp_fragment]
            if ref_fragment in id2phrase:
                ref_fragment = id2phrase[ref_fragment]
            diff_vocab_after_spellcheck_vs_ref[(tag, hyp_fragment, ref_fragment)] += 1

        # here we loop over changed fragments from end to begin to ensure correct slicing
        for tag, hyp_fragment, ref_fragment, i1, i2, j1, j2 in reversed(get_changed_fragments(before_words, ref_words)):
            if ref_fragment in id2phrase:
                ideal_words = ideal_words[:i1] + [ref_fragment] + ideal_words[i2:]
                out_ideal.write("\t\t" + hyp_fragment + "->" + id2phrase[ref_fragment] + "\n")

            if hyp_fragment in id2phrase:
                hyp_fragment = id2phrase[hyp_fragment]
            if ref_fragment in id2phrase:
                ref_fragment = id2phrase[ref_fragment]

            old_frag = " " + hyp_fragment + " "
            new_frag = " " + ref_fragment + " "
            if new_frag in after_sent:
                diff_vocab_before_spellcheck_vs_ref[(tag, hyp_fragment, ref_fragment, "!")] += 1
            else:
                diff_vocab_before_spellcheck_vs_ref[(tag, hyp_fragment, ref_fragment, "-")] += 1

        ideal_spellchecked_sent = " " + " ".join(ideal_words) + " "
        for phrase_id in id2phrase:
            ideal_spellchecked_sent = ideal_spellchecked_sent.replace(" " + phrase_id + " ", " " + id2phrase[phrase_id] + " ")

        out_ideal.write("orig:  " + " ".join(before_words) + "\n")
        out_ideal.write("ideal: " + ideal_spellchecked_sent + "\n")
        out_ideal.write("ref : " + ref_sent + "\n")

        for phrase_id in id2phrase:
            ref_sent = ref_sent.replace(" " + phrase_id + " ", " " + id2phrase[phrase_id] + " ")
            before_sent = before_sent.replace(" " + phrase_id + " ", " " + id2phrase[phrase_id] + " ")
            after_sent = after_sent.replace(" " + phrase_id + " ", " " + id2phrase[phrase_id] + " ")

        for tag, hyp_fragment, ref_fragment, i1, i2, j1, j2 in get_changed_fragments(before_words, after_words):
            if hyp_fragment in id2phrase:
                hyp_fragment = id2phrase[hyp_fragment]
            if ref_fragment in id2phrase:
                ref_fragment = id2phrase[ref_fragment]
            old_frag = " " + hyp_fragment + " "
            new_frag = " " + ref_fragment + " "
            if new_frag in ref_sent:
                diff_vocab_before_after_spellcheck[(tag, hyp_fragment, ref_fragment, "!")] += 1
            elif old_frag in ref_sent:
                diff_vocab_before_after_spellcheck[(tag, hyp_fragment, ref_fragment, "*")] += 1
            else:
                diff_vocab_before_after_spellcheck[(tag, hyp_fragment, ref_fragment, "?")] += 1


        test_data[i]["pred_text"] = ideal_spellchecked_sent.strip()
        test_data[i]["before_spell_pred"] = before_sent.strip()  # we need this field for "ideal" manifest even if it did not existed before 

        out.write(doc_id + "\t" + "ref_sent=\t" + ref_sent + "\n")
        out.write(doc_id + "\t" + "after_sent=\t" + after_sent + "\n")
        for phrase in vocab:
            phr = " " + phrase + " " 
            if phr in ref_sent:
                correct = "-"
                if phr in before_sent and not phr in after_sent:
                    correct = "*"
                if phr not in before_sent and phr in after_sent:
                    correct = "!"
                if phr in before_sent and phr in after_sent:
                    correct = "+"
                out.write("\t" + correct + "\t" + phrase + "\n")
                if correct == "!":
                    out_target_better.write(correct + "\t" + doc_id + "\t" + phrase + "\n " + after_sent + "\n " + before_sent + "\n " + ref_sent + "\n")
                    count_better += 1
                elif correct == "*":
                    out_target_worse.write(correct + "\t" + doc_id + "\t" + phrase + "\n " + after_sent + "\n " + before_sent + "\n " + ref_sent + "\n")
                    count_worse += 1
                elif correct == "-":
                    out_target_lost.write(correct + "\t" + doc_id + "\t" + phrase + "\n " + after_sent + "\n " + ref_sent + "\n")
                    count_lost += 1
                else:
                    out_target_good.write(correct + "\t" + doc_id + "\t" + phrase + "\n " + after_sent + "\n")
                    count_good += 1
            else:   
                # phr not in ref sent! false positive
                if phr not in before_sent and phr in after_sent:
                    correct = "#"
                    out_target_fp.write(correct + "\t" + doc_id + "\t" + phrase + "\n " + after_sent + "\n " + before_sent + "\n " + ref_sent + "\n")
                    count_fp += 1
out_target_lost.close()
out_target_worse.close()
out_target_better.close()
out_target_good.close()
out_target_fp.close()
out_ideal.close()

write_manifest(args.output_name + ".ideal_spellcheck", test_data)

print("Lost:", count_lost)
print("Worse:", count_worse)
print("Better:", count_better)
print("Good: ", count_good)
print("Fp: ", count_fp)

sum = 0
print("AFTER SPELLCHECKING vs REF")
for k, v in diff_vocab_after_spellcheck_vs_ref.most_common(1000000):
    sum += v
    print(k, v, "sum=", sum)

sum = 0
print("BEFORE SPELLCHECKING vs REF")
for k, v in diff_vocab_before_spellcheck_vs_ref.most_common(1000000):
    sum += v
    print(k, v, "sum=", sum)

sum_better = 0
sum_worse = 0
sum_unknown = 0
print("BEFORE vs AFTER SPELLCHECKING")
for k, v in diff_vocab_before_after_spellcheck.most_common(1000000):
    if k[3] == "!":
        sum_better += v
    elif k[3] == "*":
        sum_worse += v
    else:
        sum_unknown += v
    print(k, v, "sum_better=", sum_better, "; sum_worse=", sum_worse, "; sum_unknown=", sum_unknown)
