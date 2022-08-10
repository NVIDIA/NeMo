# ! /usr/bin/python
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from pathlib import Path


def load_data(manifest: Path):
    data = []
    with manifest.open("r") as f:
        for line in f.readlines():
            item = json.loads(line)
            data.append(item)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests_dir", default="", help="manifests directory")
    args = parser.parse_args()

    manifests = Path(args.manifests_dir).glob("*_slu.json")
    cnt = 0
    for manifest in manifests:
        print(f"Processing file: {manifest}")
        split = manifest.stem.split("_")[0]
        output_file = Path(manifest.parent) / Path(f"{split}_asr.json")
        data = load_data(manifest)
        with output_file.open("w") as f:
            for item in data:
                if not item["transcript"]:
                    cnt += 1
                    continue
                item["text"] = item["transcript"]
                f.write(f"{json.dumps(item)}\n")
        print(f"Saved output to: {output_file}, {cnt} items discarded.")
