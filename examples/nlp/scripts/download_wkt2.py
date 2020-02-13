import os
import subprocess

from nemo import logging


def download_wkt2(data_dir):
    if os.path.exists(data_dir):
        logging.warning(f'Folder {data_dir} found. Skipping download.')
        return
    os.makedirs(os.path.join(data_dir, 'lm'), exist_ok=True)
    logging.warning(f'Data not found at {data_dir}. Downloading wikitext-2 to {data_dir}/lm/')
    data_dir = 'data/lm/wikitext-2'
    subprocess.call('get_wkt2.sh')
    return data_dir
