# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
This script uses multi-processing to add normalized text into manifest files.

$ python <nemo_root_path>/scripts/dataset_processing/tts/add_normalized_text.py \
    --src=<data_root_path>/fastpitch_manifest.json \
    --dst=<data_root_path>/fastpitch_manifest.json \
    --src-key=text \
    --dst-key=normalized_text
"""
import argparse
import multiprocessing
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.append(f"{BASE_DIR}")

def get_normalizer():
    with tempfile.TemporaryDirectory() as data_dir:
        # data_dir = BASE_DIR / "data" / "normalizer"
        # data_dir.mkdir(parents=True, exist_ok=False)

        normalizer = Normalizer(
            lang="en",
            input_case="cased",
            whitelist=None,
            overwrite_cache=True,
            cache_dir=None,  # str(data_dir / "tts_cache_dir"),
        )
    return normalizer


def normalize(text):
    text_normalizer_call_kwargs = {"verbose": False, "punct_pre_process": True, "punct_post_process": True}
    return normalizer.normalize(text, **text_normalizer_call_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="original manifest")
    parser.add_argument("--dst", type=str, help="path to save manifest")
    parser.add_argument("--src-key", type=str, default="text")
    parser.add_argument("--dst-key", type=str, default="normalized_text")
    args = parser.parse_args()

    records: List[Dict[str, Any]] = read_manifest(args.src)

    # there is a problem with picking normalizer object
    # you can avoid global var by passing normalizer to each call of normalize(...)
    # but it will be ~1.5x slower, than current approach with global variable
    global normalizer
    normalizer = get_normalizer()

    text_key = args.src_key
    text_normalized_key = args.dst_key
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        texts_normalized = list(tqdm(p.imap(normalize, [record[text_key] for record in records]), total=len(records)))

    for record, text_normalized in zip(records, texts_normalized):
        record[text_normalized_key] = text_normalized

    write_manifest(args.dst, records)


if __name__ == "__main__":
    main()
