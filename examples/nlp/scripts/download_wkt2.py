import argparse
import os
import subprocess

from nemo import logging


def _download_wkt2(data_dir):
    if os.path.exists(data_dir):
        logging.warning(f'Folder {data_dir} found. Skipping download.')
        return
    os.makedirs(data_dir, exist_ok=True)
    logging.warning(f'Downloading wikitext-2 to {data_dir}')
    subprocess.call(['./get_wkt2.sh', data_dir])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download wikitext-2 dataset')
    parser.add_argument("--data_dir", required=True, type=str)

    args = parser.parse_args()
    _download_wkt2(args.data_dir)
