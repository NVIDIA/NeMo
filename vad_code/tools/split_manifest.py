import json
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("manifest", type=str, help="manifest file to process")
parser.add_argument("-r", "--ratio", type=float, default=0.05, help="ratio for validation")
parser.add_argument("-s", "--subsample", type=float, default=1.0, help="ratio for subsampling the dataset")
parser.add_argument("-o", "--output", default="", help="output dir")


def load_manifest(filepath: str):
    data = []
    durations = 0.0
    with Path(filepath).open("r") as fin:
        for line in fin.readlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            data.append(item)
            durations += item["duration"]
    return data, durations


def get_durations(data):
    durations = 0.0
    for item in data:
        durations += item["duration"]
    return durations


def save_manifest(data: List[Dict], out_file: str):
    with Path(out_file).open("w") as fout:
        for item in data:
            fout.write(f"{json.dumps(item)}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    input_file = Path(args.manifest)
    if args.output == "":
        out_dir = input_file.parent
    else:
        out_dir = Path(args.output)
    ratio = float(args.ratio)

    out_file_train = out_dir / Path(f"{input_file.stem}_train.json")
    out_file_dev = out_dir / Path(f"{input_file.stem}_dev.json")

    print("Processing...")
    all_data, total_durations = load_manifest(input_file)
    np.random.shuffle(all_data)

    if args.subsample < 1.0:
        idx = int(np.ceil(len(all_data) * args.subsample))
        all_data = all_data[:idx]

    idx = int(np.ceil(len(all_data) * ratio))
    data_dev = all_data[:idx]
    data_train = all_data[idx:]

    train_duration = get_durations(data_train)
    dev_duration = get_durations(data_dev)
    save_manifest(data_train, out_file_train)
    save_manifest(data_dev, out_file_dev)
    print(
        f"Original {total_durations/3600:.2f}hr, after processing got train={train_duration/3600:.2f}hr, dev={dev_duration/3600:.2f}hr."
    )
    print("Done")
