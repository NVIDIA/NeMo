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

    with requests.get(url, stream=True) as read_file, open(save_path, "wb") as write_file:
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


def extract_single_zst_file(input_path, save_dir, file_name, rm_input=False):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    if os.path.exists(save_path):
        print(f"File {save_path} already exists, skipping extraction.")
        return save_path

    total_length = os.stat(input_path).st_size
    with tqdm.tqdm(
        total=total_length,
        unit="B",
        unit_scale=True,
        desc=file_name,
    ) as pbar:
        dctx = zstd.ZstdDecompressor()
        read_size = 131075
        write_size = int(read_size * 4)
        save_path = os.path.join(save_dir, file_name)
        update_len = 0
        with open(input_path, "rb") as in_f, open(save_path, "wb") as out_f:
            for chunk in dctx.read_to_iter(in_f, read_size=read_size, write_size=write_size):
                out_f.write(chunk)
                update_len += read_size
                if update_len >= 3000000:
                    pbar.update(update_len)
                    update_len = 0
    if rm_input:
        os.remove(input_path)


def convert_file_numbers(file_numbers_str):
    final_list = []
    split_comma = file_numbers_str.split(",")
    for elem in split_comma:
        if elem == "":
            continue
        if "-" in elem:
            split_dash = elem.split("-")
            final_list += list(range(int(split_dash[0]), int(split_dash[1]) + 1))
        else:
            final_list.append(int(elem))
    return final_list


def split_list(inlist, ngroups):
    """Splits list into groups.
    inlist = list(range(18))  # given list
    ngroups = 5  # desired number of parts
    Returns: [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9], [10, 11, 12, 13],
              [14, 15, 16, 17]]
    """
    nlen = len(inlist)
    list_groups = []
    for ii in range(ngroups):
        idx_start = (ii * nlen) // ngroups
        idx_end = ((ii + 1) * nlen) // ngroups
        list_groups.append(inlist[idx_start:idx_end])
    return list_groups


def is_tool(progname):
    """Check whether `name` is on PATH and marked as executable."""
    # https://stackoverflow.com/a/34177358/3457624
    return which(progname) is not None
