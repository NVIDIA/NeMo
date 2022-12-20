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
import io
import os
import shutil
import sys
import time

from nemo_megatron.utils.file_utils import download_single_file

NEMO_MEGATRON_CI = os.getenv("NEMO_MEGATRON_CI", "False").lower() in ("true", "t", "1")
VERSIONS = ["v1.1", "v2.0", "xquad"]
VERSION2PATHS = {
    "v1.1": [
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
    ],
    "v2.0": [
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    ],
    "xquad": [
        f"https://raw.githubusercontent.com/deepmind/xquad/master/xquad.{lang}.json"
        for lang in ["en", "es", "de", "el", "ru", "tr", "ar", "vi", "th", "zh", "hi"]
    ],
}


def download_squad(data_dir, versions):
    os.makedirs(data_dir, exist_ok=True)

    for v in versions:
        if os.path.exists(os.path.join(data_dir, v)):
            print(f"Skipped downloading SQuAD {v}. Already exists.")
            # download might not finish in time in CI
            if NEMO_MEGATRON_CI:
                time.sleep(5)
            continue

        print(f"Downloading SQuAD {v}...")
        for url in VERSION2PATHS[v]:
            download_single_file(url, os.path.join(data_dir, v))
        print("\tCompleted!")


def get_versions(requested_versions):
    requested_versions = requested_versions.split(",")

    if "all" in requested_versions:
        versions = VERSIONS
    else:
        versions = []
        for v in requested_versions:
            if v.lower() in VERSIONS:
                versions.append(v)
            else:
                raise ValueError(f"SQuAD version \"{v}\" not found!")

    versions = set(versions)
    return list(versions)


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="directory to save data to", type=str, default="squad_data")
    parser.add_argument(
        "--versions",
        help="SQuAD versions (v1.1, v2.0 or xquad) to download data for as a comma separated string",
        type=str,
        default="all",
    )
    args = parser.parse_args(arguments)
    versions = get_versions(args.versions)
    download_squad(args.data_dir, versions)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
