# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
# USAGE: python get_lj_speech_data.py --data_root=<where to put data>
import argparse
import logging
import os
import tarfile
import urllib.request

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


def get_args():
    parser = argparse.ArgumentParser(description='LJSpeech Data download')
    parser.add_argument("--data_root", required=True, type=str)
    return parser.parse_args()


def __maybe_download_file(destination: str):
    """
    Downloads source to destination if it doesn't exist.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource
    Returns:
    """
    if not os.path.exists(destination):
        logging.info("{0} does not exist. Downloading ...".format(destination))
        urllib.request.urlretrieve(URL, filename=destination + '.tmp')
        os.rename(destination + '.tmp', destination)
        logging.info("Downloaded {0}.".format(destination))
    else:
        logging.info("Destination {0} exists. Skipping.".format(destination))
    return destination


def __extract_file(filepath: str, data_dir: str):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        logging.info("Not extracting. Maybe already there?")


def main(data_root):
    logging.info("\n\nWorking on LJSpeech")
    filepath = os.path.join(data_root, "LJSpeech-1.1.tar.bz2")
    logging.info("Getting LJSpeech")
    __maybe_download_file(filepath)
    logging.info("Extracting LJSpeech")
    __extract_file(filepath, data_root)
    logging.info('Done!')


if __name__ == "__main__":
    args = get_args()
    main(args.data_root)
