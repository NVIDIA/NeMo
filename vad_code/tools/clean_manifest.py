import json
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import librosa
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("manifest", type=str, help="manifest file to process")
parser.add_argument("-rd", "--remote_dir", default="", help="prefix dir from remote server")
parser.add_argument("-ld", "--local_dir", default="", help="prefix dir on local machine")
parser.add_argument("-o", "--output", default="", help="output filepath")

args = parser.parse_args()


def load_manifest(filepath: str):
    data = []
    with Path(filepath).open("r") as fin:
        for line in fin.readlines():
            data.append(json.loads(line.strip()))
    return data


def save_manifest(data: List[Dict], out_file: str):
    with Path(out_file).open("w") as fout:
        for item in data:
            fout.write(f"{json.dumps(item)}\n")


def check_data_sanity(data: List[Dict]):
    results = []
    for item in tqdm(data):
        audio_path = item["audio_filepath"]
        if Path(audio_path).is_file():
            try:
                # _ = librosa.load(audio_path, sr=16000)
                results.append(item)
            except:
                continue
        elif args.remote_dir != "" and args.local_dir != "":
            if audio_path.startswith(args.remote_dir):
                audio_path_local = args.local_dir + audio_path[len(args.remote_dir) :]
                if Path(audio_path_local).is_file():
                    try:
                        # _ = librosa.load(audio_path_local, sr=16000)
                        results.append(item)
                    except:
                        continue
    return results


def main():
    print(f"Processing manifest: {args.manifest}")
    data = load_manifest(args.manifest)
    data_cleaned = check_data_sanity(data)
    diff = len(data) - len(data_cleaned)
    print(f"{diff} files were removed")
    if args.output == "":
        out_file = args.manifest.split(".")[0] + "_cleaned.json"
    else:
        out_file = args.output

    print(f"Saving output to: {out_file}")
    save_manifest(data_cleaned, out_file)
    print("Done!")


if __name__ == "__main__":
    main()
