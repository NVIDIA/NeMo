# This script is used to combine several TTS datasets together for use in training.
# It requires you to have run the get_data.py associated with each dataset that you want to use.
# Simply feed in the training, validation and test jsons associated with each dataset and this 
# script will combine the jsons as needed.
# None of the arguments are required so you can choose to combine only the train or only the val sets if you wish to.
# Usage:
# python data_combiner.py --output_dir <output dir> --train <list of train manifests> --val <list of val manifests> --test <list of test manifests>

import argparse
import json
import os
import random
from pathlib import Path

parser = argparse.ArgumentParser(description='Combine multiple dataset manifests')
parser.add_argument("--output_dir", required=True, type=Path)
parser.add_argument("--train", required=False, type=Path, nargs='*')
parser.add_argument("--val", required=False, type=Path, nargs='*')
parser.add_argument("--test", required=False, type=Path, nargs='*')
args = parser.parse_args()

def create_manifest(output_dir, manifests, split):
    output = []
    for manifest in manifests:
        with open(manifest, "r") as fin:
            data = fin.readlines()
            for line in data:
                output.append(json.loads(line))

    random.shuffle(output)

    with open(os.path.join(output_dir, f"combined_{split}.json"), 'w') as fout:
        for x in output:
            fout.write(json.dumps(x) + '\n')

def main():
    train_manifests = args.train
    val_manifests = args.val
    test_manifests = args.test
    output_dir = args.output_dir

    if not output_dir.exists():
        os.mkdir(str(output_dir))

    if train_manifests:
        create_manifest(output_dir, train_manifests, split="train")
        print("Train manifest created.")

    if val_manifests:
        create_manifest(output_dir, val_manifests, split="val")
        print("Validation manifest created.")

    if test_manifests:
        create_manifest(output_dir, test_manifests, split="test")
        print("Test manifest created.")

if __name__ == "__main__":
    main()
