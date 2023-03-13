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
from argparse import ArgumentParser
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    get_alignment_by_dp,
    load_ngram_mappings_for_dp,
)

parser = ArgumentParser(description="Analyze custom phrases recognition after ASR")
parser.add_argument(
    "--manifest", required=True, type=str, help="Path to manifest file with reference text and asr hypotheses"
)
parser.add_argument("--vocab_dir", type=str, required=True, help="Path to directory with custom vocabularies")
parser.add_argument(
    "--input_dir", type=str, required=True, help="Path to input directory with asr-hypotheses and candidates"
)
parser.add_argument("--ngram_mappings", type=str, required=True, help="File with ngram mappings")
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


def read_hypotheses_and_candidates(hypotheses_filename, candidates_filename):
    candidates2hypotheses = defaultdict(set)  # key=candidate, value=set of asr_hypotheses
    hypotheses_lines = []
    candidates_lines = []
    with open(hypotheses_filename, "r", encoding="utf-8") as f:
        hypotheses_lines = f.readlines()
    with open(candidates_filename, "r", encoding="utf-8") as f:
        candidates_lines = f.readlines()
    if len(hypotheses_lines) != len(candidates_lines):
        raise (ValueError, "number of lines doesn't match in " + hypotheses_filename + " and " + candidates_filename)
    for i in range(len(hypotheses_lines)):
        s = hypotheses_lines[i]
        hyp, _ = s.strip().split("\t")
        hyp = hyp.replace(" ", "").replace("_", " ")
        candidates = candidates_lines[i].strip().split(";")
        for c in candidates:
            text, begin, _, _, _ = c.split("|")
            if begin == -1:
                continue
            text = text.replace(" ", "").replace("_", " ")
            candidates2hypotheses[text].add(hyp)
    return candidates2hypotheses


joint_vocab, src_vocab, dst_vocab, max_len = load_ngram_mappings_for_dp(args.ngram_mappings)

custom_vocabs = {}  # key=doc_id, value=tuple(set of phrases, phrase2id, id2phrase)
for name in os.listdir(args.vocab_dir):
    parts = name.split(".")
    if len(parts) == 3 and parts[1] == "custom":
        doc_id = parts[0]
        custom_vocabs[doc_id] = read_custom_vocab(os.path.join(args.vocab_dir, name))

# this is used to check if lost phrase was ever proposed as candidate
candidates2hypotheses = {}  # key=doc_id, value=dict(key=candidate, value=set of asr_hypotheses)
for name in os.listdir(args.input_dir):
    parts = name.split(".")
    if len(parts) == 2:
        doc_id = parts[0]
        candidates2hypotheses[doc_id] = read_hypotheses_and_candidates(
            os.path.join(args.input_dir, doc_id + ".txt"), os.path.join(args.input_dir, doc_id + ".info.txt"),
        )

test_data = read_manifest(args.manifest)
pred_text = [data['pred_text'] for data in test_data]
ref_text = [data['text'] for data in test_data]
audio_filepath = [data['audio_filepath'] for data in test_data]
doc_ids = []
for data in test_data:
    if "doc_id" in data:
        doc_ids.append(data["doc_id"])
    else:  # fix for Spoken Wikipedia format
        path = data["audio_filepath"]
        # example of path: ...clips/197_0000.wav   #doc_id=197
        path_parts = path.split("/")
        path_parts2 = path_parts[-1].split("_")
        doc_id = path_parts2[-2]
        doc_ids.append(doc_id)

out_target_lost = open(args.output_name + ".target_lost", "w", encoding="utf-8")
out_target_lost_before_top10 = open(args.output_name + ".target_lost_before_top10", "w", encoding="utf-8")
out_target_lost_after_top10 = open(args.output_name + ".target_lost_after_top10", "w", encoding="utf-8")
out_target_worse = open(args.output_name + ".target_worse", "w", encoding="utf-8")
out_target_better = open(args.output_name + ".target_better", "w", encoding="utf-8")
out_target_good = open(args.output_name + ".target_good", "w", encoding="utf-8")
out_target_fp = open(args.output_name + ".target_fp", "w", encoding="utf-8")
out_ideal = open(args.output_name + ".ideal", "w", encoding="utf-8")

count_lost = 0
count_lost_before_top10 = 0
count_lost_after_top10 = 0
count_worse = 0
count_better = 0
count_good = 0
count_fp = 0

with open(args.output_name, "w", encoding="utf-8") as out:
    diff_vocab_after_spellcheck_vs_ref = Counter()
    diff_vocab_before_spellcheck_vs_ref = Counter()
    diff_vocab_before_after_spellcheck = Counter()

    for i in range(len(test_data)):
        doc_id = doc_ids[i]
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
        for tag, hyp_fragment, ref_fragment, i1, i2, j1, j2 in reversed(
            get_changed_fragments(before_words, ref_words)
        ):
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
            ideal_spellchecked_sent = ideal_spellchecked_sent.replace(
                " " + phrase_id + " ", " " + id2phrase[phrase_id] + " "
            )

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
                verdict = "!"
            elif old_frag in ref_sent:
                verdict = "*"
            else:
                verdict = "?"
            diff_vocab_before_after_spellcheck[(tag, hyp_fragment, ref_fragment, verdict)] += 1

        test_data[i]["pred_text"] = ideal_spellchecked_sent.strip()
        # we need this field for "ideal" manifest even if it did not existed before
        test_data[i]["before_spell_pred"] = before_sent.strip()

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
                    out_target_better.write(
                        correct
                        + "\t"
                        + doc_id
                        + "\t"
                        + phrase
                        + "\n "
                        + after_sent
                        + "\n "
                        + before_sent
                        + "\n "
                        + ref_sent
                        + "\n"
                    )
                    count_better += 1
                elif correct == "*":
                    out_target_worse.write(
                        correct
                        + "\t"
                        + doc_id
                        + "\t"
                        + phrase
                        + "\n "
                        + after_sent
                        + "\n "
                        + before_sent
                        + "\n "
                        + ref_sent
                        + "\n"
                    )
                    count_worse += 1
                elif correct == "-":
                    out_target_lost.write(
                        correct + "\t" + doc_id + "\t" + phrase + "\n " + after_sent + "\n " + ref_sent + "\n"
                    )
                    count_lost += 1
                    # check whether the lost phrase ever got to top10 candidates or not
                    found = False
                    if phrase in candidates2hypotheses[doc_id]:
                        for hyp in candidates2hypotheses[doc_id][phrase]:
                            if hyp in before_sent:
                                found = True
                                break
                    if found:
                        count_lost_after_top10 += 1
                        out_target_lost_after_top10.write(
                            correct + "\t" + doc_id + "\t" + phrase + "\n " + after_sent + "\n " + ref_sent + "\n"
                        )
                    else:
                        count_lost_before_top10 += 1
                        out_target_lost_before_top10.write(
                            correct + "\t" + doc_id + "\t" + phrase + "\n " + after_sent + "\n " + ref_sent + "\n"
                        )
                else:
                    out_target_good.write(correct + "\t" + doc_id + "\t" + phrase + "\n " + after_sent + "\n")
                    count_good += 1
            else:
                # phr not in ref sent! false positive
                if phr not in before_sent and phr in after_sent:
                    correct = "#"
                    out_target_fp.write(
                        correct
                        + "\t"
                        + doc_id
                        + "\t"
                        + phrase
                        + "\n "
                        + after_sent
                        + "\n "
                        + before_sent
                        + "\n "
                        + ref_sent
                        + "\n"
                    )
                    count_fp += 1
out_target_lost.close()
out_target_lost_before_top10.close()
out_target_lost_after_top10.close()
out_target_worse.close()
out_target_better.close()
out_target_good.close()
out_target_fp.close()
out_ideal.close()

write_manifest(args.output_name + ".ideal_spellcheck", test_data)

print("Lost:", count_lost)
print("Lost before top10:", count_lost_before_top10)
print("Lost after top10:", count_lost_after_top10)
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
    path = get_alignment_by_dp(
        " ".join(list(k[2].replace(" ", "_"))),
        " ".join(list(k[1].replace(" ", "_"))),
        joint_vocab,
        src_vocab,
        dst_vocab,
        max_len,
    )
    if k[3] == "!":
        sum_better += v
    elif k[3] == "*":
        sum_worse += v
    else:
        sum_unknown += v

    print(
        k,
        v,
        "sum_better=",
        sum_better,
        "; sum_worse=",
        sum_worse,
        "; sum_unknown=",
        sum_unknown,
        "; av_score=",
        path[-1][3] / (0.001 + len(k[1])),
    )
    for hyp_ngram, ref_ngram, score, sum_score, joint_freq, src_freq, dst_freq in path:
        print(
            "\t",
            "hyp=",
            hyp_ngram,
            "; ref=",
            ref_ngram,
            "; score=",
            score,
            "; sum_score=",
            sum_score,
            joint_freq,
            src_freq,
            dst_freq,
        )
