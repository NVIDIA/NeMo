# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
    manifests = expand_braces(manifests)
    with open(final_cache_manifest, 'w', encoding='utf-8') as cache_f:
        for manifest in manifests:
            with open(manifest, 'r') as m:
                for line in m.readlines():
                    data_entry = json.loads(line)
                    json.dump(data_entry, cache_f, ensure_ascii=False)
                    cache_f.write('\n')


def handle_multiple_tarr_filepaths(mmanifest_file: str, tmpdir: str, number_of_manifests: int, tarr_file: str):
    base_manifest_name = mmanifest_file.rsplit('_', 1)[0]
    rank = torch.distributed.get_rank()

    start_range = rank * number_of_manifests
    end_range = start_range + number_of_manifests - 1

    temporary_manifest = os.path.join(tmpdir, f"temp_{base_manifest_name}_{{{start_range}..{end_range}}}.json")
    base_path, tarr_filename = os.path.split(tarr_file)
    base_tarr_name = tarr_filename.rsplit('_', 1)[0]

    expanded_audio_path = os.path.join(base_path, f"{base_tarr_name}_{{{start_range}..{end_range}}}.tar")

    return temporary_manifest, expanded_audio_path


def write_tarr_cache_manifest(
    cache_manifests: str,
    update_data: List[Dict],
    hypotheses: List,
    update_size: int = 0,
    indices: Optional[List] = None,
    use_lhotse: bool = False,
):
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
    cache_manifest: str, hypotheses: List[str], data: List[Dict], update_whole_cache: bool = True,
):
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
    regex_punctuation = re.compile(fr"([{''.join(punctuation)}])")
    regex_extra_space = re.compile('\s{2,}')
    return regex_extra_space.sub(' ', regex_punctuation.sub(' ', line)).strip()


def process_manifest(manifest_path):
    manifest_data = []
    with open(manifest_path, 'r', encoding='utf_8') as manifest_file:
        for line in manifest_file:
            data_entry = json.loads(line)
            manifest_data.append(data_entry)
    return manifest_data


def sample_data(data, weight, update_whole_cache, p_cache):
    weight_factor = weight * p_cache if not update_whole_cache else weight
    sample_size = int(len(data) * weight_factor / torch.distributed.get_world_size())
    return random.sample(data, sample_size)
