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
#
# Code adapted from https://github.com/speechbrain/speechbrain/blob/develop/recipes/SLURP/prepare.py


import argparse
import json
import os
from pathlib import Path
from typing import Optional

import jsonlines
import pandas as pd
from tqdm import tqdm


def prepare_SLURP(
    data_folder: str, save_folder: Optional[str] = None, join_semantics: bool = True, audio_prefix: str = "",
):
    """
    This function prepares the SLURP dataset.
    If the folder does not exist, the zip file will be extracted. If the zip file does not exist, it will be downloaded.
    Params:
    -   data_folder: path to SLURP dataset.
    -   save_folder: path where to save the csv manifest files.
    -   join_semantics: whether to convert semantics to string representation
    -   audio_prefix: prefix added to audio paths
    """

    if not save_folder:
        save_folder = data_folder
    elif not Path(save_folder).is_dir():
        Path(save_folder).mkdir(parents=True)

    splits = [
        "train",
        "train_synthetic",
        "devel",
        "test",
    ]
    id = 0
    for split in splits:
        new_filename = os.path.join(save_folder, split) + f"_slu.json"
        print(f"Preparing {new_filename}...")

        IDs = []
        slurp_id = []
        audio = []
        audio_format = []
        audio_opts = []

        semantics = []
        semantics_format = []
        semantics_opts = []

        transcript = []
        transcript_format = []
        transcript_opts = []

        jsonl_path = os.path.join(data_folder, split + ".jsonl")

        with jsonlines.open(jsonl_path) as reader:
            for obj in reader:
                sid = obj["slurp_id"]
                scenario = obj["scenario"]
                action = obj["action"]
                sentence_annotation = obj["sentence_annotation"]
                num_entities = sentence_annotation.count("[")
                entities = []
                for slot in range(num_entities):
                    type = sentence_annotation.split("[")[slot + 1].split("]")[0].split(":")[0].strip()
                    filler = sentence_annotation.split("[")[slot + 1].split("]")[0].split(":")[1].strip()
                    entities.append({"type": type.lower(), "filler": filler.lower()})
                for recording in obj["recordings"]:
                    IDs.append(id)
                    slurp_id.append(sid)
                    if "synthetic" in split:
                        audio_folder = "slurp_synth/"
                    else:
                        audio_folder = "slurp_real/"

                    path = os.path.join(audio_prefix, audio_folder, recording["file"])

                    audio.append(path)
                    audio_format.append("flac")
                    audio_opts.append(None)

                    transcript.append(obj["sentence"])
                    transcript_format.append("string")
                    transcript_opts.append(None)

                    semantics_dict = {
                        "scenario": scenario,
                        "action": action,
                        "entities": entities,
                    }
                    if join_semantics:
                        semantics_ = str(semantics_dict)
                    else:
                        semantics_ = semantics_dict
                    semantics.append(semantics_)
                    semantics_format.append("string")
                    semantics_opts.append(None)
                    id += 1

        df = pd.DataFrame(
            {"ID": IDs, "slurp_id": slurp_id, "audio": audio, "semantics": semantics, "transcript": transcript,}
        )

        with open(new_filename, "w") as fout:
            for idx in tqdm(range(len(df))):
                item = {
                    "id": str(df["ID"][idx]),
                    "slurp_id": str(df["slurp_id"][idx]),
                    "audio_filepath": df["audio"][idx],
                    "transcript": df["transcript"][idx],
                    "semantics": df["semantics"][idx],
                    "text": df["semantics"][idx],
                }
                fout.write(json.dumps(item) + "\n")
        print(f"Saved output to: {new_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_root", type=str, default="slurp_data/", help="Data directory containing the raw annotation files"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="slurp_data/", help="Output directory containing the processed files"
    )
    parser.add_argument(
        "-js",
        "--join_semantics",
        type=bool,
        default=True,
        help="whether to convert semantics dict to str (default=True)",
    )

    args = parser.parse_args()
    prepare_SLURP(data_folder=args.data_root, save_folder=args.output, join_semantics=args.join_semantics)
