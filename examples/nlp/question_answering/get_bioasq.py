#!/bin/bash
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
# Disclaimer:
# All the data in this repository is no longer updated since 2019.Jan.24th and it may not reflect current data available.
#
#### BioASQ
# Before using the files in this repository, you must first register BioASQ website and download the [BioASQ Task B](http://participants-area.bioasq.org/Tasks/A/getData/) data.
# See "An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition (Tsatsaronis et al. 2015)" for datasets details.
#
# Copyright 2019 dmis-lab/biobert.
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
import logging
import os
import subprocess
from distutils.dir_util import copy_tree

from nemo import logging

URL = {
    'bioasq': 'https://drive.google.com/uc?id=19ft5q44W4SuptJgTwR84xZjsHg1jvjSZ',
    'bioasq_6b': 'https://drive.google.com/uc?id=1-KzAQzaE-Zd4jOlZG_7k7D4odqPI3dL1',
}


def download(download_url: str, parent_dir: str):
    tmp_zip = '/tmp/data.zip'
    tmp_unzip = '/tmp/data'
    if not os.path.exists(tmp_unzip):
        os.makedirs(tmp_unzip)
    else:
        subprocess.run(['rm', '-rf', tmp_unzip])
    subprocess.run(['gdown', '-O', tmp_zip, download_url])
    subprocess.run(['unzip', tmp_zip, '-d', tmp_unzip])
    copy_tree(tmp_unzip, parent_dir)
    subprocess.run(['rm', '-rf', tmp_zip])
    subprocess.run(['rm', '-rf', tmp_unzip])


def __maybe_download_file(parent_dir: str):
    """
    from https://github.com/dmis-lab/biobert download https://drive.google.com/uc?id=19ft5q44W4SuptJgTwR84xZjsHg1jvjSZ
    from https://github.com/dmis-lab/bioasq-biobert#datasets  https://drive.google.com/uc?id=1-KzAQzaE-Zd4jOlZG_7k7D4odqPI3dL1

    If exists, skips download
    Args:
        parent_dir: local filepath
    """
    target_dir = os.path.join(parent_dir, 'BioASQ')
    if os.path.exists(target_dir):
        logging.info(f'{target_dir} found. Skipping download')
    else:
        download_url = URL['bioasq']
        logging.info(f'Downloading {download_url} from https://github.com/dmis-lab/biobert to {target_dir}')
        download(download_url, parent_dir)
    parent_dir = target_dir
    target_dir = os.path.join(parent_dir, 'BioASQ-6b')
    if os.path.exists(target_dir):
        logging.info(f'{target_dir} found. Skipping download')
    else:
        download_url = URL['bioasq_6b']
        logging.info(
            f'Downloading {download_url} from https://github.com/dmis-lab/bioasq-biobert#datasets to {target_dir}'
        )
        download(download_url, parent_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument("--data_dir", required=True, type=str, help="directory to download dataset to")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    logging.info(f'Downloading dataset')
    __maybe_download_file(args.data_dir)
