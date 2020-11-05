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

import os
from pathlib import Path
from time import sleep

import wget

from nemo.utils import logging


def maybe_download_from_cloud(url, filename, subfolder=None, cache_dir=None, refresh_cache=False) -> str:
    """
    Helper function to download pre-trained weights from the cloud
    Args:
        url: (str) URL of storage
        filename: (str) what to download. The request will be issued to url/filename
        subfolder: (str) subfolder within cache_dir. The file will be stored in cache_dir/subfolder. Subfolder can
            be empty
        cache_dir: (str) a cache directory where to download. If not present, this function will attempt to create it.
            If None (default), then it will be $HOME/.cache/torch/NeMo
        refresh_cache: (bool) if True and cached file is present, it will delete it and re-fetch

    Returns:
        If successful - absolute local path to the downloaded file
        else - empty string
    """
    # try:
    if cache_dir is None:
        cache_location = Path.joinpath(Path.home(), ".cache/torch/NeMo")
    else:
        cache_location = cache_dir
    if subfolder is not None:
        destination = Path.joinpath(cache_location, subfolder)
    else:
        destination = cache_location

    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)

    destination_file = Path.joinpath(destination, filename)

    if os.path.exists(destination_file):
        logging.info(f"Found existing object {destination_file}.")
        if refresh_cache:
            logging.info("Asked to refresh the cache.")
            logging.info(f"Deleting file: {destination_file}")
            os.remove(destination_file)
        else:
            logging.info(f"Re-using file from: {destination_file}")
            return str(destination_file)
    # download file
    wget_uri = url + filename
    logging.info(f"Downloading from: {wget_uri} to {str(destination_file)}")
    # NGC links do not work everytime so we try and wait
    i = 0
    max_attempts = 3
    while i < max_attempts:
        i += 1
        try:
            wget.download(wget_uri, str(destination_file))
            if os.path.exists(destination_file):
                return destination_file
            else:
                return ""
        except:
            logging.info(f"Download from cloud failed. Attempt {i} of {max_attempts}")
            sleep(0.05)
            continue
    raise ValueError("Not able to download url right now, please try again.")
