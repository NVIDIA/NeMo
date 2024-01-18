# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from kaldialign import align
from typing import List, Optional, Dict, Union


def load_data(manifest: str) -> List[Dict]:
    """
    Load data from manifest file.

    Args:
        manifest: path to nemo manifest file.
    Returns:
        List of dicts with keys: audio_filepath, text, pred_text.
    """
    data = []
    with open(manifest, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data
    

def update_stats(item_ref, item_hyp, key_words_stat):
        if item_ref in key_words_stat:
            key_words_stat[item_ref][1] += 1 # add to totall
            if item_ref == item_hyp:
                key_words_stat[item_ref][0] += 1 # add to tp
        elif item_hyp in key_words_stat:
            key_words_stat[item_hyp][2] += 1 # add to fp


def compute_fscore(recognition_results_manifest: str, key_words_list: List, return_scores: bool = False) -> Optional[tuple]:
    """
    Compute fscore for list of context biasing words/phrases.
    The idea is to get a word-level alignment for ground truth text and prediction results from manifest file.
    Then compute f-score for each word/phrase from key_words_list according to obtained word alignment.

    Args:
        recognition_results_manifest: path to nemo manifest file with recognition results in pred_text field.
        key_words_list: list of context biasing words/phrases.
        return_scores: if True, return precision, recall and fscore (not only print).
    Returns:
        If return_scores is True, return tuple of precision, recall and fscore.
    """

    # get data from manifest
    data = load_data(recognition_results_manifest)
    # compute max number of words in one context biasing phrase
    max_ngram_order = max([len(item.split()) for item in key_words_list])
    key_words_stat = {} # a word here can be single word or phareses 
    for word in key_words_list:
        key_words_stat[word] = [0, 0, 0] # [true positive (tp), groud truth (gt), false positive (fp)]

    # auxiliary variable for epsilon token during alignment 
    eps = '***'

    for item in data:
        ref = item['text'].split()
        hyp = item['pred_text'].split()
        ali = align(ref, hyp, eps)

        for idx, pair in enumerate(ali):

            # check all the ngrams:
            # TODO -- add epsilon skipping to ge more accurate results for phrases...
            for ngram_order in range(1, max_ngram_order+1):
                if (idx+ngram_order-1) < len(ali):
                    item_ref, item_hyp = [], []
                    for order in range(1, ngram_order+1):
                        item_ref.append(ali[idx+order-1][0])
                        item_hyp.append(ali[idx+order-1][1])
                    item_ref = " ".join(item_ref)
                    item_hyp = " ".join(item_hyp)
                    update_stats(item_ref, item_hyp, key_words_stat)
                else:
                    break
    

    tp = sum([key_words_stat[x][0] for x in key_words_stat])
    gt = sum([key_words_stat[x][1] for x in key_words_stat])
    fp = sum([key_words_stat[x][2] for x in key_words_stat])

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (gt + 1e-8)
    fscore = 2*(precision*recall)/(precision+recall + 1e-8)

    print("\n"+"***"*15)
    print("Per words statistic (word: correct/totall | false positive):\n")
    max_len = max([len(x) for x in key_words_stat if key_words_stat[x][1] > 0 or key_words_stat[x][2] > 0])
    for word in key_words_list:
        if key_words_stat[word][1] > 0 or key_words_stat[word][2] > 0:
            false_positive = ""
            if key_words_stat[word][2] > 0:
                false_positive = key_words_stat[word][2]
            print(f"{word:>{max_len}}: {key_words_stat[word][0]:3}/{key_words_stat[word][1]:<3} |{false_positive:>3}")
    print("***"*20)
    print(" ")
    print("***"*10)
    print(f"Precision: {precision:.4f} ({tp}/{tp + fp}) fp:{fp}")
    print(f"Recall:    {recall:.4f} ({tp}/{gt})")
    print(f"Fscore:    {fscore:.4f}")
    print("***"*10)

    if return_scores:
        return (precision, recall, fscore)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_manifest", type=str, required=True, help="nemo manifest with recognition results in pred_text field",
    )
    parser.add_argument(
        "--context_biasing_file", type=str, required=True, help="file of context biasing words/phrases with their spellings"
    )

    args = parser.parse_args()
    # use list instead of dict to preserve key words order during printing word-level statistics
    key_words_list = []
    for line in open(args.context_biasing_file).readlines():
        item = line.strip().split("-")[0].lower()
        if item not in key_words_list:
            key_words_list.append(item)
    compute_fscore(args.input_manifest, key_words_list)


if __name__ == '__main__':
    main()