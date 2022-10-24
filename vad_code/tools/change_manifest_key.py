import argparse
import json
from ast import parse
from copy import deepcopy
from pathlib import Path


def load_manifest(filepath):
    data = []
    with Path(filepath).open("r") as fin:
        for line in fin.readlines():
            data.append(json.loads(line.strip()))
    return data


def save_manifest(data, filepath):
    with Path(filepath).open("w") as fout:
        for item in data:
            fout.write(f"{json.dumps(item)}\n")


def change_key(data, src_key, tgt_key):
    results = []
    for item in data:
        item = deepcopy(item)
        if tgt_key in item:
            print("already has target key, skipping...")
            break
        if src_key not in item:
            for key in item.keys():
                if src_key in key:
                    print(f"replacing source key {src_key} with new key {key}")
                    src_key = key
                    break
        item[tgt_key] = item[src_key]
        item.pop(src_key)
        results.append(item)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="path to manifest to be processed")
    parser.add_argument("-s", "--source", default="vad_mask", help="the original key you want to change")
    parser.add_argument("-t", "--target", default="label", help="the target key you want to change to")
    args = parser.parse_args()

    filepath = Path(args.manifest)
    if filepath.is_dir():
        manifest_list = list(filepath.glob("*.json"))
        print(f"Found {len(manifest_list)} files to be processed.")
    else:
        manifest_list = [filepath]

    for manifest in manifest_list:
        print(f"Processing: {manifest}")
        data = load_manifest(manifest)
        new_data = change_key(data, args.source, args.target)

        print(f"Saving output to: {manifest}")
        save_manifest(new_data, manifest)

    print("Done!")
