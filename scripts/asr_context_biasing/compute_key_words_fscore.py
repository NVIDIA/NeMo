#!/usr/bin/env python

import argparse
import json
import os
from kaldialign import align


def load_data(manifest):
    data = []
    with open(manifest, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data


def print_alignment(audio_filepath, ali, key_words):
    ref, hyp = [], []
    for pair in ali:
        if pair[1] in key_words:
            ref.append(pair[0].upper()) 
            hyp.append(pair[1].upper()) 
        else:
            ref.append(pair[0])
            hyp.append(pair[1])
    print(" ")
    print(f"ID: {os.path.basename(audio_filepath)}")
    print(f"REF: {' '.join(ref)}")
    print(f"HYP: {' '.join(hyp)}")
    

def update_stats(item_ref, item_hyp, key_words_stat):
        if item_ref in key_words_stat:
            key_words_stat[item_ref][1] += 1 # add to totall
            if item_ref == item_hyp:
                key_words_stat[item_ref][0] += 1 # add to tp
        elif item_hyp in key_words_stat:
            key_words_stat[item_hyp][2] += 1 # add to fp


def compute_fscore(recognition_results_manifest, key_words_list, print_ali=False, return_scores=False):

    data = load_data(recognition_results_manifest)
    max_ngram_order = max([len(item.split()) for item in key_words_list])
    key_words_stat = {} # a word here can be single word or phareses 
    for word in key_words_list:
        key_words_stat[word] = [0, 0, 0] # [tp, totall, fp]

    eps = '***'

    for item in data:
        audio_filepath = item['audio_filepath']
        ref = item['text'].split()
        hyp = item['pred_text'].split()
        ali = align(ref, hyp, eps)
        false_positive_words = []

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
                    
        # if recognized_words and print_ali:
        #     # print_alignment(audio_filepath, ali, recognized_words)
        #     print_alignment(audio_filepath, ali, false_positive_words)

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
    print("***"*15)
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
        "--input_manifest", type=str, required=True, help="manifest with recognition results",
    )
    parser.add_argument(
        "--key_words_file", type=str, required=True, help="file of key words for fscore calculation"
    )

    args = parser.parse_args()
    #key_words_list = [x for x in args.key_words_list.split('_')]
    key_words_list = []
    for line in open(args.key_words_file).readlines():
        item = line.strip().split("-")[0].lower()
        # item = line.strip().lower()
        if item not in key_words_list:
            key_words_list.append(item)
    compute_fscore(args.input_manifest, key_words_list)


if __name__ == '__main__':
    main()