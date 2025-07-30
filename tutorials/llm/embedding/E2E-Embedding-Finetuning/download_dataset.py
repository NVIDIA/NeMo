#!/usr/bin/env python3
"""
download_allnli_triplet.py

This script downloads the 'triplet' subset of the sentence-transformers/all-nli
dataset from Hugging Face Datasets (specifically the train split) and converts
each entry into a dictionary with the following structure:

{
  "query":   <anchor sentence>,
  "pos_doc": <positive sentence>,
  "neg_doc": <negative sentence>
}

The result is saved as a JSON file: `allnli_triplet_train.json`
"""

import json
from datasets import load_dataset

def main():
    # Step 1: Load the 'triplet' subset of the all-nli dataset (train split)
    print("Downloading dataset...")
    ds = load_dataset('sentence-transformers/all-nli', 'triplet', split='train')

    # Step 2: Transform each example into a dictionary with query, pos_doc, and neg_doc
    print("Processing records...")
    records = [{
        "query":   example["anchor"],
        "pos_doc": example["positive"],
        "neg_doc": example["negative"],
    } for example in ds]

    # Step 3: Save the processed dataset as a pretty-printed JSON file
    out_path = "allnli_triplet.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f" Saved {len(records)} triplets to {out_path}")

if __name__ == "__main__":
    main()

