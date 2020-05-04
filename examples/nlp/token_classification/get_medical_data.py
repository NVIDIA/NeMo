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
    'bc5cdr': 'https://drive.google.com/uc?id=1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh',
    'ncbi': 'https://drive.google.com/uc?id=1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh',
}


def __maybe_download_file(destination: str, dataset: str):
    """
    Downloads data from https://github.com/dmis-lab/biobert#datasets named entity recognition to destination if not exists.
    If exists, skips download
    Args:
        destination: local filepath
        dataset: dataset
    """
    parent_source, child_source = dataset.split("-")
    download_url = URL[parent_source]
    if not os.path.exists(destination):
        logging.info(f'Downloading {download_url} from https://github.com/dmis-lab/biobert#datasets to {destination}')
        tmp_zip = '/tmp/data.zip'
        tmp_unzip = '/tmp/data'
        if not os.path.exists(tmp_unzip):
            os.makedirs(tmp_unzip)
        else:
            subprocess.run(['rm', '-rf', tmp_unzip])
        subprocess.run(['gdown', '-O', tmp_zip, download_url])
        subprocess.run(['unzip', tmp_zip, '-d', tmp_unzip])

        subprocess.run(['mv', os.path.join(tmp_unzip, f"{parent_source.upper()}-{child_source}"), destination])
        if os.path.exists(os.path.join(destination, "devel.tsv")):
            subprocess.run(['mv', os.path.join(destination, "devel.tsv"), os.path.join(destination, "dev.tsv")])
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
