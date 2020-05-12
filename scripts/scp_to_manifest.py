# Copyright 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os

import librosa as l
from tqdm import tqdm


def main(scp, id, out):
    if os.path.exists(out):
        os.remove(out)
    scp_file = open(scp, 'r').readlines()

    with open(out, 'w') as outfile:
        for line in tqdm(scp_file):
            line = line.strip()
            y, sr = l.load(line, sr=None)
            dur = l.get_duration(y=y, sr=sr)
            speaker = line.split('/')[id]
            speaker = list(speaker)
            # speaker[0]='P'
            speaker = ''.join(speaker)
            # outfile.write("{}  {:.3f} {}\n".format(line,dur,speaker))
            meta = {"audio_filepath": line, "duration": float(dur), "label": speaker}
            json.dump(meta, outfile)
            outfile.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scp", help="scp file name", type=str)
    parser.add_argument("--id", help="field num seperated by '/' to be considered as speaker label", type=int)
    parser.add_argument("--out", help="manifest_file name", type=str)
    args = parser.parse_args()

    main(args.scp, args.id, args.out)
