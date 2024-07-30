# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import braceexpand
import torch
from omegaconf import ListConfig


def expand_braces(filepaths):
    """
    Expands brace expressions in file paths.

    Args:
        filepaths (str or ListConfig): The file path(s) to expand.

    Returns:
        list: A list of expanded file paths.
    """
    if isinstance(filepaths, ListConfig):
        filepaths = filepaths[0]

    if isinstance(filepaths, str):
        # Replace '(' and '[' with '{'
        brace_keys_open = ['(', '[', '<', '_OP_']
        for bkey in brace_keys_open:
            if bkey in filepaths:
                filepaths = filepaths.replace(bkey, "{")

        # Replace ')' and ']' with '}'
        brace_keys_close = [')', ']', '>', '_CL_']
        for bkey in brace_keys_close:
            if bkey in filepaths:
                filepaths = filepaths.replace(bkey, "}")

    if isinstance(filepaths, str):
        # Brace expand, set escape=False for Windows compatibility
        filepaths = list(braceexpand.braceexpand(filepaths, escape=False))
    return filepaths


def formulate_cache_manifest_names(manifests: Union[str, ListConfig[str]], cache_prefix, is_tarred: bool):
    """
    Formulates cache manifest names based on the provided manifests and cache prefix.

    Args:
        manifests (Union[str, ListConfig[str]]): The original manifest file paths. If tarred,
            this should be a list of lists of manifest file paths.
        cache_prefix (str): The prefix to use for the cache manifest names.
        is_tarred (bool): A flag indicating whether the dataset is tarred.

    Returns:
        Union[str, List[List[str]]]: The cache manifest names. If the dataset is tarred,
            returns a list of lists of cache manifest names. Otherwise, returns a single cache
            manifest name as a string.

    """
    if is_tarred:
        cache_manifests = []
        if isinstance(manifests, str):
            manifests = [[manifests]]
        for sharded_manifests in manifests:
            base_path, file_name = os.path.split(sharded_manifests[0])
            cache_name = os.path.join(base_path, f'{cache_prefix}_cache_{file_name}')
            cache_manifests.append([cache_name])
        return cache_manifests
    else:
        return str(Path.cwd() / f"{cache_prefix}_pseudo_labeled.json")


def count_files_for_pseudo_labeling(
    input_manifest_files: Union[str, ListConfig[str]],
    is_tarred: bool,
    dataset_weights: Optional[Union[float, ListConfig[float]]] = None,
) -> Tuple[List[int], List[int]]:
    """
    Counts the number of files for pseudo-labeling based on the input manifest files.

    Args:
        input_manifest_files (Union[str, ListConfig[str]]): The manifest files containing
            the dataset information. Can be a single file path or a list of file paths.
        is_tarred (bool): A flag indicating whether the dataset is tarred.
        dataset_weights (Optional[Union[float, ListConfig[float]]]): Weights for the datasets.
            If not provided, defaults to 1 for each dataset. This option works only for non tarr datasets.

    Returns:
        Tuple[List[int], List[int]]: A tuple containing two lists:
            - The number of files in each manifest.
            - The weighted number of files for pseudo-labeling based on the dataset weights.

    """

    def get_num_lines(file_path: str) -> int:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return len(file.readlines())
        return 0

    if is_tarred:
        dataset_weights = None
        all_manifests = []
        if isinstance(input_manifest_files, str):
            all_manifests = expand_braces(input_manifest_files)
        else:
            for tarr_manifest in input_manifest_files:
                all_manifests += expand_braces(tarr_manifest)
    else:
        all_manifests = input_manifest_files

    if not dataset_weights:
        dataset_weights = [1] * len(all_manifests)
    if not isinstance(dataset_weights, ListConfig) and not isinstance(dataset_weights, List):
        dataset_weights = [float(dataset_weights)]

    if isinstance(all_manifests, str):
        num_all_files = get_num_lines(all_manifests)
        return [num_all_files], [int(num_all_files * dataset_weights[0])]
    else:
        num_all_files = [get_num_lines(manifest_path) for manifest_path in all_manifests]
        num_cache_files = [
            int(files * dataset_weights[idx]) if idx < len(dataset_weights) else files
            for idx, files in enumerate(num_all_files)
        ]
        return num_all_files, num_cache_files


def create_final_cache_manifest(final_cache_manifest: str, manifests: List[str]):
    """
    Creates a final cache manifest by combining multiple manifest files into one.

    Args:
        final_cache_manifest (str): The path to the final cache manifest file to be created.
        manifests (List[str]): A list of manifest file paths to be combined into the final cache manifest.

    """
    manifests = expand_braces(manifests)
    with open(final_cache_manifest, 'w', encoding='utf-8') as cache_f:
        for manifest in manifests:
            with open(manifest, 'r') as m:
                for line in m.readlines():
                    data_entry = json.loads(line)
                    json.dump(data_entry, cache_f, ensure_ascii=False)
                    cache_f.write('\n')


def handle_multiple_tarr_filepaths(manifest_file: str, tmpdir: str, number_of_manifests: int, tarr_file: str):
    """
    Handles multiple tarred file paths by generating temporary manifest and expanded audio paths.

    Args:
        manifest_file (str): The base manifest file name.
        tmpdir (str): The directory for storing temporary files.
        number_of_manifests (int): The number of manifest files to handle.
        tarr_file (str): The base tarred file path.

    Returns:
        Tuple[str, str]: A tuple containing the temporary manifest path and the expanded audio path.

    """
    base_manifest_name = manifest_file.rsplit('_', 1)[0]
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    start_range = rank * number_of_manifests
    end_range = start_range + number_of_manifests - 1

    temporary_manifest = os.path.join(tmpdir, f"temp_{base_manifest_name}_{{{start_range}..{end_range}}}.json")
    base_path, tarr_filename = os.path.split(tarr_file)
    base_tarr_name = tarr_filename.rsplit('_', 1)[0]

    expanded_audio_path = os.path.join(base_path, f"{base_tarr_name}_{{{start_range}..{end_range}}}.tar")
    if "pipe:ais" in base_path:
        expanded_audio_path += " -"
    return temporary_manifest, expanded_audio_path


def write_tar_cache_manifest(
    cache_manifests: str,
    update_data: List[Dict],
    hypotheses: List,
    update_size: int = 0,
    indices: Optional[List] = None,
    use_lhotse: bool = False,
):
    """
    Writes the tarred cache manifest files with updated data entries.

    Args:
        cache_manifests (str): Paths to the cache manifest files.
        update_data (List[Dict]): Data entries to be updated.
        hypotheses (List): List of hypotheses to be added to the data entries.
        update_size (int, optional): Size of the update batch. Defaults to 0.
        indices (Optional[List], optional): Indices of the data entries to be updated. Defaults to None.
        use_lhotse (bool, optional): Flag indicating whether we use Lhotse dataloaders.
            In that case tar manifest files should not be shuffled.

    """
    if update_size == 0:
        for i, chache_file in enumerate(cache_manifests):
            with open(chache_file, 'w', encoding='utf-8') as cache_f:
                for j, data_entry in enumerate(update_data[i]):
                    data_entry['text'] = hypotheses[i * len(update_data[0]) + j]
                    json.dump(data_entry, cache_f, ensure_ascii=False)
                    cache_f.write('\n')
    else:
        j = 0
        for i in range(len(update_data)):
            for idx in indices[i]:
                update_data[i][idx]['text'] = hypotheses[j]
                j += 1
        if not use_lhotse:
            for i in range(len(update_data)):
                random.shuffle(update_data[i])
        for i, chache_file in enumerate(cache_manifests):
            with open(chache_file, 'w', encoding='utf-8') as cache_f:
                for j, data_entry in enumerate(update_data[i]):
                    json.dump(data_entry, cache_f, ensure_ascii=False)
                    cache_f.write('\n')


def write_cache_manifest(
    cache_manifest: str,
    hypotheses: List[str],
    data: List[Dict],
    update_whole_cache: bool = True,
):
    """
    Writes the cache manifest file with updated data entries.

    Args:
        cache_manifest (str): Path to the cache manifest file.
        hypotheses (List[str]): List of hypotheses to be added to the data entries.
        data (List[Dict]): Data entries to be updated.
        update_whole_cache (bool, optional): Flag indicating whether we updated the whole cache.
            If False, the cache is shuffled and only the specified entries are updated. Defaults to True.

    """
    if update_whole_cache:
        with open(cache_manifest, 'w', encoding='utf-8') as cache_file:
            for i, audio_data in enumerate(data):
                for j, data_entry in enumerate(audio_data):
                    data_entry['text'] = hypotheses[i][j]
                    json.dump(data_entry, cache_file, ensure_ascii=False)
                    cache_file.write('\n')
    else:
        cache_data = []
        update_size = len(data) * len(data[0])
        if update_size > 0:
            with open(cache_manifest, 'r', encoding='utf-8') as cache_file:
                for i, line in enumerate(cache_file):
                    data_entry = json.loads(line)
                    cache_data += [data_entry]
            random.shuffle(cache_data)
            with open(cache_manifest, 'w', encoding='utf-8') as cache_file:
                for data_entry in cache_data[:-update_size]:
                    json.dump(data_entry, cache_file, ensure_ascii=False)
                    cache_file.write('\n')
                for i, audio_data in enumerate(data):
                    for j, data_entry in enumerate(audio_data):
                        data_entry['text'] = hypotheses[i][j]
                        json.dump(data_entry, cache_file, ensure_ascii=False)
                        cache_file.write('\n')


def rm_punctuation(line: str, punctuation: str):
    """
    Removes specified punctuation from a line of text and replaces multiple spaces with a single space.

    Args:
        line (str): The input text line from which to remove punctuation.
        punctuation (str): A string of punctuation characters to be removed.

    Returns: The text line with punctuation removed and extra spaces replaced.

    """
    regex_punctuation = re.compile(fr"([{''.join(punctuation)}])")
    regex_extra_space = re.compile('\s{2,}')
    return regex_extra_space.sub(' ', regex_punctuation.sub(' ', line)).strip()


def process_manifest(manifest_path):
    """
    Reads and processes a manifest file, returning its data entries as a list.

    Args:
        manifest_path (str): The path to the manifest file.

    Returns:
        list: A list of data entries from the manifest file.
    """
    manifest_data = []
    with open(manifest_path, 'r', encoding='utf_8') as manifest_file:
        for line in manifest_file:
            data_entry = json.loads(line)
            manifest_data.append(data_entry)
    return manifest_data


def sample_data(data, weight, update_whole_cache, p_cache):
    """
    Samples a subset of data based on the given weight and cache parameters.

    Args:
        data (list): The input data to sample from.
        weight (float): The weight factor to determine the sample size.
        update_whole_cache (bool): Flag indicating whether the whole cache is being updated.
        p_cache (float): The cache percentage to be used if not updating the whole cache.

    Returns:
        list: A subset of the input data sampled based on the calculated sample size.

    """
    weight_factor = weight * p_cache if not update_whole_cache else weight
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    sample_size = int(len(data) * weight_factor / world_size)
    return random.sample(data, sample_size)
