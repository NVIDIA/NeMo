# Copyright (c) 2020, MeetKai Inc.  All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
This script downloads Text2Sparql data and processes it into NeMo's neural machine translation dataset format.

Text2Sparql data consists of 3 files which are saved to source_data_dir:
    - train_queries_v3.tsv
    - test_easy_queries_v3.tsv
    - test_hard_queries_v3.tsv

After processing, the script saves them to the target_data_dir as:
    - train.tsv
    - test_easy.tsv
    - test_hard.tsv


You may run it with:

python import_datasets \
    --source_data_dir ./text2sparql_src \
    --target_data_dir ./text2sparql_tgt
"""

import argparse
import csv
import os
from urllib.request import Request, urlopen

from nemo.collections.nlp.data.data_utils.data_preprocessing import MODE_EXISTS_TMP, if_exist
from nemo.utils import logging

base_url = "https://m.meetkai.com/public_datasets/knowledge/"
prefix_map = {
    "train_queries_v3.tsv": "train.tsv",
    "test_easy_queries_v3.tsv": "test_easy.tsv",
    "test_hard_queries_v3.tsv": "test_hard.tsv",
}


def download_text2sparql(infold: str):
    """Downloads text2sparql train, test_easy, and test_hard data

    Args:
        infold: save directory path
    """
    os.makedirs(infold, exist_ok=True)

    for prefix in prefix_map:
        url = base_url + prefix

        logging.info(f"Downloading: {url}")
        if if_exist(infold, [prefix]):
            logging.info("** Download file already exists, skipping download")
        else:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with open(os.path.join(infold, prefix), "wb") as handle:
                handle.write(urlopen(req, timeout=20).read())


def process_text2sparql(infold: str, outfold: str, do_lower_case: bool):
    """ Process and convert MeetKai's text2sparql datasets to NeMo's neural machine translation format.

    Args:
        infold: directory path to raw text2sparql data containing
            train.tsv, test_easy.tsv, test_hard.tsv
        outfold: output directory path to save formatted data for NeuralMachineTranslationDataset
            the first line is header (sentence [tab] label)
            each line should be [sentence][tab][label]
        do_lower_case: if true, convert all sentences and labels to lower
    """
    logging.info(f"Processing Text2Sparql dataset and storing at: {outfold}")

    os.makedirs(outfold, exist_ok=True)

    dataset_name = "Text2Sparql"
    for prefix in prefix_map:
        input_file = os.path.join(infold, prefix)
        output_file = os.path.join(outfold, prefix_map[prefix])

        if if_exist(outfold, [prefix_map[prefix]]):
            logging.info(f"** {MODE_EXISTS_TMP.format(prefix_map[prefix], dataset_name, output_file)}")
            continue

        if not if_exist(infold, [prefix]):
            logging.info(f"** {prefix} of {dataset_name}" f" is skipped as it was not found")
            continue

        assert input_file != output_file, "input file cannot equal output file"
        with open(input_file, "r") as in_file:
            with open(output_file, "w") as out_file:
                reader = csv.reader(in_file, delimiter="\t")

                # replace headers
                out_file.write("sentence\tlabel\n")
                next(reader)

                for line in reader:
                    sentence = line[0]
                    label = line[1]
                    if do_lower_case:
                        sentence = sentence.lower()
                        label = label.lower()
                    out_file.write(f"{sentence}\t{label}\n")


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
