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


import argparse
import json
import os
from collections import defaultdict

from tqdm.auto import tqdm

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    get_alignment_by_dp,
    load_ngram_mappings_for_dp,
)

parser = argparse.ArgumentParser()
parser.add_argument("--asr_hypotheses_folder", required=True, type=str, help="Input folder with asr hypotheses")
parser.add_argument(
    "--spellchecker_inputs_folder",
    required=True,
    type=str,
    help="Input folder with spellchecker inputs, here .info.txt files are needed",
)
parser.add_argument(
    "--spellchecker_results_folder", required=True, type=str, help="Input folder with spellchecker output"
)
parser.add_argument("--input_manifest", required=True, type=str, help="Manifest with transcription before correction")
parser.add_argument("--output_manifest", required=True, type=str, help="Manifest with transcription after correction")
parser.add_argument("--min_cov", required=True, type=float, help="Minimum coverage value")
parser.add_argument("--min_real_cov", required=True, type=float, help="Minimum real coverage value")
parser.add_argument(
    "--min_dp_score_per_symbol",
    required=True,
    type=float,
    help="Minimum dynamic programming sum score averaged by hypothesis length",
)
parser.add_argument("--min_dst_len", default=1, type=int, help="Minimum dst length")
parser.add_argument("--ngram_mappings", type=str, required=True, help="File with ngram mappings")

args = parser.parse_args()


def read_manifest(path):
    manifest = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest


def check_banned_replacements(src, dst):
    if src.endswith(" l") and dst.endswith(" l y") and src[0:-2] == dst[0:-4]:
        return True
    if dst.endswith(" l") and src.endswith(" l y") and dst[0:-2] == src[0:-4]:
        return True
    if src.endswith(" l") and dst.endswith(" l l y") and src[0:-2] == dst[0:-6]:
        return True
    if dst.endswith(" l") and src.endswith(" l l y") and dst[0:-2] == src[0:-6]:
        return True
    if src.endswith(" e") and dst.endswith(" e s") and src[0:-2] == dst[0:-4]:
        return True
    if dst.endswith(" e") and src.endswith(" e s") and dst[0:-2] == src[0:-4]:
        return True
    if src.endswith(" e") and dst.endswith(" a l") and src[0:-2] == dst[0:-4]:
        return True
    if dst.endswith(" e") and src.endswith(" a l") and dst[0:-2] == src[0:-4]:
        return True
    if src.endswith(" a") and dst.endswith(" a n") and src[0:-2] == dst[0:-4]:
        return True
    if dst.endswith(" a") and src.endswith(" a n") and dst[0:-2] == src[0:-4]:
        return True
    if src.endswith(" e") and dst.endswith(" y") and src[0:-2] == dst[0:-2]:
        return True
    if dst.endswith(" e") and src.endswith(" y") and dst[0:-2] == src[0:-2]:
        return True
    if src.endswith(" i e s") and dst.endswith(" y") and src[0:-6] == dst[0:-2]:
        return True
    if dst.endswith(" i e s") and src.endswith(" y") and dst[0:-6] == src[0:-2]:
        return True
    if src.endswith(" i e s") and dst.endswith(" y ' s") and src[0:-6] == dst[0:-6]:
        return True
    if dst.endswith(" i e s") and src.endswith(" y ' s") and dst[0:-6] == src[0:-6]:
        return True
    if src.endswith(" e") and dst.endswith(" i n g") and src[0:-2] == dst[0:-6]:
        return True
    if dst.endswith(" e") and src.endswith(" i n g") and dst[0:-2] == src[0:-6]:
        return True
    if src.endswith(" e s") and dst.endswith(" i n g") and src[0:-4] == dst[0:-6]:
        return True
    if dst.endswith(" e s") and src.endswith(" i n g") and dst[0:-4] == src[0:-6]:
        return True
    if src.endswith(" c e s") and dst.endswith(" x") and src[0:-6] == dst[0:-2]:
        return True
    if dst.endswith(" c e s") and src.endswith(" x") and dst[0:-6] == src[0:-2]:
        return True
    if src.endswith(" e") and dst.endswith(" e d") and src[0:-2] == dst[0:-4]:
        return True
    if dst.endswith(" e") and src.endswith(" e d") and dst[0:-2] == src[0:-4]:
        return True
    if src.endswith(" e") and dst.endswith(" i c") and src[0:-2] == dst[0:-4]:
        return True
    if dst.endswith(" e") and src.endswith(" i c") and dst[0:-2] == src[0:-4]:
        return True
    if src.endswith(" y") and dst.endswith(" i c") and src[0:-2] == dst[0:-4]:
        return True
    if dst.endswith(" y") and src.endswith(" i c") and dst[0:-2] == src[0:-4]:
        return True
    if src.endswith(" n") and dst.endswith(" n y") and src[0:-2] == dst[0:-4]:
        return True
    if dst.endswith(" n") and src.endswith(" n y") and dst[0:-2] == src[0:-4]:
        return True
    if src.endswith(" e d") and dst.endswith(" i n g") and src[0:-4] == dst[0:-6]:
        return True
    if dst.endswith(" e d") and src.endswith(" i n g") and dst[0:-4] == src[0:-6]:
        return True
    if src.endswith(" e n c e") and dst.endswith(" i n g") and src[0:-8] == dst[0:-6]:
        return True
    if dst.endswith(" e n c e") and src.endswith(" i n g") and dst[0:-8] == src[0:-6]:
        return True
    if src.endswith(" a n c e") and dst.endswith(" i n g") and src[0:-8] == dst[0:-6]:
        return True
    if dst.endswith(" a n c e") and src.endswith(" i n g") and dst[0:-8] == src[0:-6]:
        return True
    if src.endswith(" e d") and dst.endswith(" e s") and src[0:-4] == dst[0:-4]:
        return True
    if dst.endswith(" e d") and src.endswith(" e s") and dst[0:-4] == src[0:-4]:
        return True
    if src.endswith(" _ i s") and dst.endswith(" e s") and src[0:-6] == dst[0:-4]:
        return True
    if dst.endswith(" _ i s") and src.endswith(" e s") and dst[0:-6] == src[0:-4]:
        return True
    if src.endswith(" _ i s") and dst.endswith(" ' s") and src[0:-6] == dst[0:-4]:
        return True
    if dst.endswith(" _ i s") and src.endswith(" ' s") and dst[0:-6] == src[0:-4]:
        return True
    if src.endswith(" _ i s") and dst.endswith(" s") and src[0:-6] == dst[0:-2]:
        return True
    if dst.endswith(" _ i s") and src.endswith(" s") and dst[0:-6] == src[0:-2]:
        return True
    if src.endswith(" _ a s") and dst.endswith(" ' s") and src[0:-6] == dst[0:-4]:
        return True
    if dst.endswith(" _ a s") and src.endswith(" ' s") and dst[0:-6] == src[0:-4]:
        return True
    if src.endswith(" _ h a s") and dst.endswith(" ' s") and src[0:-8] == dst[0:-4]:
        return True
    if dst.endswith(" _ h a s") and src.endswith(" ' s") and dst[0:-8] == src[0:-4]:
        return True
    if src.endswith(" t i o n") and dst.endswith(" t e d") and src[0:-8] == dst[0:-6]:
        return True
    if dst.endswith(" t i o n") and src.endswith(" t e d") and dst[0:-8] == src[0:-6]:
        return True
    if src.endswith(" t i o n") and dst.endswith(" t i v e") and src[0:-8] == dst[0:-8]:
        return True
    if dst.endswith(" t i o n") and src.endswith(" t i v e") and dst[0:-8] == src[0:-8]:
        return True
    if src.endswith(" s m") and dst.endswith(" s t s") and src[0:-4] == dst[0:-6]:
        return True
    if dst.endswith(" s m") and src.endswith(" s t s") and dst[0:-4] == src[0:-6]:
        return True
    if src.endswith(" s m") and dst.endswith(" s t") and src[0:-4] == dst[0:-4]:
        return True
    if dst.endswith(" s m") and src.endswith(" s t") and dst[0:-4] == src[0:-4]:
        return True

    if src.endswith(" '") and src[0:-2] == dst:
        return True
    if dst.endswith(" '") and dst[0:-2] == src:
        return True
    if src.endswith(" ' s") and dst.endswith(" s") and src[0:-4] == dst[0:-2]:
        return True
    if dst.endswith(" ' s") and src.endswith(" s") and dst[0:-4] == src[0:-2]:
        return True
    if src.endswith(" ' s") and dst.endswith(" e s") and src[0:-4] == dst[0:-4]:
        return True
    if dst.endswith(" ' s") and src.endswith(" e s") and dst[0:-4] == src[0:-4]:
        return True
    if src.endswith(" ' s") and dst.endswith(" y") and src[0:-4] == dst[0:-2]:
        return True
    if dst.endswith(" ' s") and src.endswith(" y") and dst[0:-4] == src[0:-2]:
        return True
    if src.endswith(" ' s") and src[0:-4] == dst:
        return True
    if dst.endswith(" ' s") and dst[0:-4] == src:
        return True
    if src.endswith(" s") and src[0:-2] == dst:
        return True
    if dst.endswith(" s") and dst[0:-2] == src:
        return True
    if src.endswith(" e d") and src[0:-4] == dst:
        return True
    if dst.endswith(" e d") and dst[0:-4] == src:
        return True

    if src.startswith("i n _ ") and src[6:] == dst:
        return True
    if dst.startswith("i n _ ") and dst[6:] == src:
        return True
    if src.startswith("o n _ ") and src[6:] == dst:
        return True
    if dst.startswith("o n _ ") and dst[6:] == src:
        return True
    if src.startswith("o f _ ") and src[6:] == dst:
        return True
    if dst.startswith("o f _ ") and dst[6:] == src:
        return True
    if src.startswith("a t _ ") and src[6:] == dst:
        return True
    if dst.startswith("a t _ ") and dst[6:] == src:
        return True

    if src.startswith("u n ") and src[4:] == dst:
        return True
    if dst.startswith("u n ") and dst[4:] == src:
        return True

    if src.startswith("r e ") and src[4:] == dst:
        return True
    if dst.startswith("r e ") and dst[4:] == src:
        return True

    return False


joint_vocab, src_vocab, dst_vocab, max_len = load_ngram_mappings_for_dp(args.ngram_mappings)

final_corrections = defaultdict(str)
banned_count = 0
for name in os.listdir(args.spellchecker_results_folder):
    doc_id, _ = name.split(".")
    short2full_sent = defaultdict(list)
    full_sent2corrections = defaultdict(dict)
    try:
        with open(args.asr_hypotheses_folder + "/" + doc_id + ".txt", "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                short_sent, full_sent = s.split("\t")
                short_sent = " ".join(list(short_sent.replace(" ", "_")))
                full_sent = " ".join(list(full_sent.replace(" ", "_")))
                short2full_sent[short_sent].append(full_sent)
        print("len(short2full_sent)=", len(short2full_sent))
    except:
        continue

    short2info = defaultdict(dict)
    input_lines = []
    info_lines = []
    with open(args.spellchecker_inputs_folder + "/" + doc_id + ".txt", "r", encoding="utf-8") as f:
        for line in f:
            input_lines.append(line.strip())
    with open(args.spellchecker_inputs_folder + "/" + doc_id + ".info.txt", "r", encoding="utf-8") as f:
        for line in f:
            info_lines.append(line.strip())
    if len(input_lines) != len(info_lines):
        raise (
            IndexError,
            "len(input_lines) != len(info_lines): " + str(len(input_lines)) + " vs " + str(len(info_lines)),
        )
    for inpline, infoline in zip(input_lines, info_lines):
        short_sent = inpline.split("\t")[0]
        info_parts = infoline.split(";")
        for part in info_parts:
            phrase, begin, length, cov, real_cov = part.split("|")
            begin = int(begin)
            length = int(length)
            cov = float(cov)
            real_cov = float(real_cov)
            short2info[short_sent][phrase] = (begin, length, cov, real_cov)

    with open(args.spellchecker_results_folder + "/" + doc_id + ".txt", "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("REPLACE"):
                continue
            parts = s.split("\t")
            _, src, dst, short_sent = parts
            if short_sent not in short2full_sent:
                continue
            if check_banned_replacements(src, dst):
                print("!!!", src, " => ", dst)
                banned_count += 1
                continue
            begin, length, cov, real_cov = short2info[short_sent][dst]
            if begin == -1:  # dummy candidate
                banned_count += 1
                continue
            if cov < args.min_cov:
                banned_count += 1
                continue
            if real_cov < args.min_real_cov:
                banned_count += 1
                continue

            if dst.count(" ") + 1 < args.min_dst_len:
                continue

            # replace hyphens with space: this fix is only for evaluation purposes (because references are without hyphens)
            dst = dst.replace("-", "_")

            for full_sent in short2full_sent[short_sent]:  # mostly there will be one-to-one correspondence
                if full_sent not in full_sent2corrections:
                    full_sent2corrections[full_sent] = {}
                if src not in full_sent2corrections[full_sent]:
                    full_sent2corrections[full_sent][src] = {}
                if dst not in full_sent2corrections[full_sent][src]:
                    full_sent2corrections[full_sent][src][dst] = 0
                full_sent2corrections[full_sent][src][dst] += 1

    print("len(full_sent2corrections)=", len(full_sent2corrections))

    for full_sent in full_sent2corrections:
        corrected_full_sent = full_sent
        for src in full_sent2corrections[full_sent]:
            for dst, freq in sorted(
                full_sent2corrections[full_sent][src].items(), key=lambda item: item[1], reverse=True
            ):
                path = get_alignment_by_dp(dst, src, joint_vocab, src_vocab, dst_vocab, max_len)
                if path[-1][3] / (src.count(" ") + 1) < args.min_dp_score_per_symbol:  # sum_score
                    continue
                corrected_full_sent = corrected_full_sent.replace(src, dst)
                # take only best variant
                break
        original_full_sent = "".join(full_sent.split()).replace(
            "_", " "
        )  # restore original format instead of separate letters
        corrected_full_sent = "".join(corrected_full_sent.split()).replace(
            "_", " "
        )  # restore original format instead of separate letters
        final_corrections[doc_id + "\t" + original_full_sent] = corrected_full_sent


print("len(final_corrections)=", len(final_corrections))

test_data = read_manifest(args.input_manifest)

# extract just the text corpus from the manifest
pred_text = [data['pred_text'] for data in test_data]
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

for i in range(len(test_data)):
    sent, doc_id = pred_text[i], doc_ids[i]
    k = doc_id + "\t" + sent
    if k in final_corrections:
        test_data[i]["before_spell_pred"] = test_data[i]["pred_text"]
        test_data[i]["pred_text"] = final_corrections[k]

with open(args.output_manifest, "w", encoding="utf-8") as out:
    for d in test_data:
        line = json.dumps(d)
        out.write(line + "\n")

print("banned count=", banned_count)
