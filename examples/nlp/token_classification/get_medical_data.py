# =============================================================================
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
# =============================================================================


import argparse
import logging
import os
import subprocess

from nemo import logging

URL = {
    'bc5cdr': 'https://github.com/ncbi-nlp/BLUE_Benchmark/releases/download/0.1/bert_data.zip',
    'ncbi': 'https://drive.google.com/uc?id=1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh',
}


def __maybe_download_file(destination: str, dataset: str):
    """
    Downloads source to destination if not exists.
    If exists, skips download
    Args:
        destination: local filepath
        dataset: dataset
    """
    parent_source, child_source = dataset.split("-")
    download_url = URL[parent_source]
    if not os.path.exists(destination):
        logging.info(f'Downloading {download_url} to {destination}')
        tmp_zip = '/tmp/data.zip'
        tmp_unzip = '/tmp/data'
        if not os.path.exists(tmp_unzip):
            os.makedirs(tmp_unzip)
        else:
            subprocess.run(['rm', '-rf', tmp_unzip])
        if parent_source == "bc5cdr":
            subprocess.run(['wget', '-O', tmp_zip, download_url])
        elif parent_source == "ncbi":
            subprocess.run(['gdown', '-O', tmp_zip, download_url])
        subprocess.run(['unzip', tmp_zip, '-d', tmp_unzip])

        if parent_source == "bc5cdr":
            subprocess.run(
                ['mv', os.path.join(tmp_unzip, f"bert_data/{parent_source.upper()}/{child_source}"), destination]
            )
        elif parent_source == "ncbi":
            subprocess.run(['mv', os.path.join(tmp_unzip, f"{parent_source.upper()}-{child_source}"), destination])

        subprocess.run(['rm', '-rf', tmp_zip])
        subprocess.run(['rm', '-rf', tmp_unzip])
    else:
        logging.info(f'{destination} found. Skipping download')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument(
        "--dataset", default='bc5cdr-chem', choices=['bc5cdr-chem', 'bc5cdr-disease', 'ncbi-disease'], type=str
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    logging.info(f'Downloading dataset')
    data_dir = os.path.join(args.data_dir, args.dataset)
    __maybe_download_file(data_dir, args.dataset)
