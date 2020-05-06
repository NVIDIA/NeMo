#!/home/nithin-dl1804/.conda/envs/NeMo/bin/python
import argparse
import os

import librosa as l
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scp", help="scp file name", type=str)
    parser.add_argument("--id", help="field num seperated by '/' to be considered as speaker label", type=int)
    parser.add_argument("--out", help="manifest_file name", type=str)
    args = parser.parse_args()

    if os.path.exists(args.out):
        os.remove(out)
    scp_file = open(args.scp, 'r').readlines()

    with open(out, 'w') as outfile:
        for line in tqdm(scp_file):
            line = line.strip()
            y, sr = l.load(line, sr=None)
            dur = l.get_duration(y=y, sr=sr)
            speaker = line.split('/')[args.id]
            speaker = list(speaker)
            # speaker[0]='P'
            speaker = ''.join(speaker)
            # outfile.write("{}  {:.3f} {}\n".format(line,dur,speaker))
            meta = {"audio_filepath": line, "duration": float(dur), "label": speaker}
            json.dump(meta, outfile)
            outfile.write("\n")
