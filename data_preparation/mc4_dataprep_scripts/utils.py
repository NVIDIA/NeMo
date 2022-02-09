import os
import requests
from shutil import which

import tqdm
import zstandard as zstd


def download_single_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    if os.path.exists(save_path):
        print(f"File {save_path} already exists, skipping download.")
        return save_path

    with requests.get(url, stream=True) as read_file, open(
        save_path, "wb"
    ) as write_file:
        total_length = int(read_file.headers.get("content-length"))
        with tqdm.tqdm(
            total=total_length,
            unit="B",
            unit_scale=True,
            desc=file_name,
        ) as pbar:
            update_len = 0
            for chunk in read_file.iter_content(chunk_size=8192):
                if chunk:
                    write_file.write(chunk)
                    update_len += len(chunk)
                    if update_len >= 1000000:
                        pbar.update(update_len)
                        update_len = 0
    return save_path