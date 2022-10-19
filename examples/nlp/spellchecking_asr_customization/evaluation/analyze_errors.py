import argparse
import os
import re
from tqdm.auto import tqdm
import json

from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--input_manifest", required=True, type=str, help="Manifest with trancription after correction")
parser.add_argument("--output_file", required=True, type=str, help="Output file")
args = parser.parse_args()


def read_manifest(path):
    manifest = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest

test_data = read_manifest(args.input_manifest)

better = 0
worse = 0
unknown = 0

with open(args.output_file, "w", encoding="utf-8") as out:
    for i in range(len(test_data)):
        if "before_spell_pred" in test_data[i] and test_data[i]["before_spell_pred"] != test_data[i]["pred_text"]:
            if test_data[i]["text"] == test_data[i]["pred_text"]:
                better += 1
#                out.write(test_data[i]["text"] + "\n")
#                out.write("\tbefore: " + test_data[i]["before_spell_pred"] + "\n")
#                out.write("\tafter: " + test_data[i]["pred_text"] + "\n")

            elif test_data[i]["text"] == test_data[i]["before_spell_pred"]:
                worse += 1
                out.write(test_data[i]["text"] + "\n")
                out.write("\tbefore: " + test_data[i]["before_spell_pred"] + "\n")
                out.write("\tafter: " + test_data[i]["pred_text"] + "\n")
            else:
                unknown += 1


print("better=", better)
print("worse=", worse)
print("unknown=", unknown)

    









