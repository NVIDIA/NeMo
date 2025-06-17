#!/usr/bin/env python

import argparse
import json
import os

from nemo.collections.asr.parts import context_biasing



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_manifest", type=str, required=True, help="manifest with recognition results",
    )
    parser.add_argument(
        "--key_words_file", type=str, required=True, help="file of key words for fscore calculation"
    )
    parser.add_argument(
        "--ctcws-mode", type=bool, default=False, help="whether to use ctcws mode to split the key words from transcriptions"
    )

    args = parser.parse_args()
    
    key_words_list = []
    for line in open(args.key_words_file, encoding='utf-8').readlines():
        if args.ctcws_mode:
            item = line.strip().split("_")[0].lower()
        else:
            item = line.strip().lower()
        if item not in key_words_list:
            key_words_list.append(item)

    fscore_stats = context_biasing.compute_fscore(args.input_manifest, key_words_list, print_stats=True)
    # print(f"Precision/Recall/Fscore = {fscore_stats[0]:.4f}/{fscore_stats[1]:.4f}/{fscore_stats[2]:.4f}")


if __name__ == '__main__':
    main()