# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2020, MeetKai Inc.  All rights reserved.
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
import csv
import os
from urllib.request import Request, urlopen

from nemo.collections.nlp.data.data_utils.data_preprocessing import MODE_EXISTS_TMP, if_exist
from nemo.utils import logging


def download_text2sparql(infold):
    base_url = "http://m.meetkai.com/public_datasets/knowledge/"
    download_urls = {
        base_url + "train_queries_v3.tsv": "train.tsv",
        base_url + "test_easy_queries_v3.tsv": "test_easy.tsv",
        base_url + "test_hard_queries_v3.tsv": "test_hard.tsv",
    }

    os.makedirs(source_dir, exist_ok=True)

    for url in download_urls:
        file = download_urls[url]

        logging.info(f"Downloading: {url}")
        if if_exist(infold, [file]):
            logging.info("** Download file already exists, skipping download")
        else:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with open(os.path.join(infold, file), "wb") as handle:
                handle.write(urlopen(req, timeout=20).read())


def process_text2sparql(infold, outfold, do_lower_case):
    """ Process and convert MeetKai's text2sparql datasets to NeMo's neural machine translation format.
    https://github.com/MeetKai/txt-2-sparql-gen/blob/master/out
    """
    logging.info(f"Processing Text2Sparql dataset and storing at: {outfold}")

    os.makedirs(outfold, exist_ok=True)

    def _read_tsv(input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    dataset_name = "Text2Sparql"
    modes = ["train", "test_easy", "test_hard"]
    for mode in modes:
        mode_file_name = f"{mode}.tsv"

        if if_exist(outfold, [mode_file_name]):
            logging.info(f"** {MODE_EXISTS_TMP.format(mode, dataset_name, os.path.join(outfold, mode_file_name))}")
            continue

        if not if_exist(infold, [mode_file_name]):
            logging.info(f"** {mode} mode of {dataset_name}" f" is skipped as it was not found")
            continue

        lines = _read_tsv(input_file=os.path.join(infold, mode_file_name))
        with open(os.path.join(outfold, mode_file_name), "w") as outfile:
            outfile.write("sentence\tlabel\n")
            for line in lines[1:]:  # skip header
                sentence = line[0]
                label = line[1]
                if do_lower_case:
                    sentence = sentence.lower()
                    label = label.lower()
                outfile.write(f"{sentence}\t{label}\n")


if __name__ == "__main__":
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser(description="Process and convert datasets into NeMo's format")
    parser.add_argument(
        "--source_data_dir", required=True, type=str, help="Path to the folder containing the dataset files"
    )
    parser.add_argument("--target_data_dir", required=True, type=str, help="Path to save the processed dataset")
    parser.add_argument("--do_lower_case", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_data_dir
    target_dir = args.target_data_dir
    do_lower_case = args.do_lower_case

    download_text2sparql(infold=source_dir)
    process_text2sparql(infold=source_dir, outfold=target_dir, do_lower_case=do_lower_case)
