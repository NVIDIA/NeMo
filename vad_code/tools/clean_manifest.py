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


def save_manifest(data: List[Dict], out_file: str):
    with Path(out_file).open("w") as fout:
        for item in data:
            fout.write(f"{json.dumps(item)}\n")


def check_data_sanity(data: List[Dict]):
    results = []
    durations = 0.0
    for item in tqdm(data):
        audio_path = item["audio_filepath"]
        if Path(audio_path).is_file():
            try:
                # _ = librosa.load(audio_path, sr=16000)
                results.append(item)
                durations += item["duration"]
            except:
                continue
        elif args.remote_dir != "" and args.local_dir != "":
            if audio_path.startswith(args.remote_dir):
                audio_path_local = args.local_dir + audio_path[len(args.remote_dir) :]
                if Path(audio_path_local).is_file():
                    try:
                        # _ = librosa.load(audio_path_local, sr=16000)
                        results.append(item)
                        durations += item["duration"]
                    except:
                        continue
    return results, durations


def main():
    print(f"Processing manifest: {args.manifest}")
    if Path(args.manifest).is_dir():
        manifests_list = [str(x) for x in Path(args.manifest).glob("*.json")]
    else:
        manifests_list = [args.manifest]
    print(f"Found {len(manifests_list)} manifests to be processed.")
    for manifest in manifests_list:
        print(f"Processing manifest: {manifest}")
        data, duration = load_manifest(manifest)
        data_cleaned, duration_cleaned = check_data_sanity(data)
        diff = len(data) - len(data_cleaned)
        print(f"{diff} files were removed")
        print(f"Original duration: {duration/3600:.2f}hrs, cleaned duration: {duration_cleaned/3600:.2f}hrs.")

        out_file = Path(manifest).stem + "_cleaned.json"
        if args.output != "":
            out_file = str(Path(args.output) / Path(out_file))
        else:
            out_file = str(Path(manifest).parent / Path(out_file))

        print(f"Saving output to: {out_file}")
        save_manifest(data_cleaned, out_file)

    print("Done!")


if __name__ == "__main__":
    main()
