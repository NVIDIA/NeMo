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

import re
import json
import random
import torch
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from omegaconf import ListConfig

def count_files_for_pseudo_labeling(
        input_manifest_files: Union[str, ListConfig[str]],
        dataset_weights: Optional[Union[float, ListConfig[float]]] = None
        ) -> Tuple[List[int], List[int]]:
    """
    Counts how many audio files are going to be used for pseudo labeling.
    Args:
        input_manifest_files: Path(s) to manifest file(s) containing unlabeled data 
        dataset_weights: Optional. What part of the dataset to use. Default is 1.0
    Returns:
        A tuple of two lists containing number of all audio files and number of audio files used for generating pseudo labels.
    """
    def get_num_lines(file_path: str) -> int:
        with open(file_path, 'r') as file:
            return len(file.readlines())
    
    if not dataset_weights:
        dataset_weights = [1] * len(input_manifest_files)
    if not isinstance(dataset_weights, ListConfig) and not isinstance(dataset_weights, List) :
            dataset_weights = [float(dataset_weights)]

    if isinstance(input_manifest_files, str):
        num_all_files = get_num_lines(input_manifest_files)
        return num_all_files, int(num_all_files * dataset_weights[0])
    else:
        num_all_files = [get_num_lines(manifest_path) for manifest_path in input_manifest_files]
        num_cache_files = [int(files * dataset_weights[idx] ) if idx < len(dataset_weights) else files for idx, files in enumerate(num_all_files)]
        return num_all_files, num_cache_files
    

def write_cache_manifest(cache_manifest: str, hypotheses: List[str], data: List[Dict], update_whole_cache: bool = True):
    """
    Writes a cache manifest file with new pseudo labels.
    Args:
        cache_manifest: Path to cache manifest file.       
        data: List of dictionaries containing audio file paths and their durations.
        hypotheses: Transcriptions for corresponding data.
        update_whole_cache: Whether to write whole cache or only update part of it.
    
    """
    if update_whole_cache:
        with open(cache_manifest, 'w', encoding='utf-8') as cache_file:
            for i,audio_data in enumerate(data):
                for j,data_entry in enumerate(audio_data):
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
                for i,audio_data in enumerate(data):
                    for j,data_entry in enumerate(audio_data):
                        data_entry['text'] = hypotheses[i][j]
                        json.dump(data_entry, cache_file, ensure_ascii=False)
                        cache_file.write('\n')

def rm_punctuation(line: str, punctuation: str):
    """
    Removes punctuation from line
    Args:
        line: String from which punctuation will be removed.
        punctuation: String containing all kind of punctuations to remove.
    """
    regex_punctuation = re.compile(fr"([{''.join(punctuation)}])")
    regex_extra_space = re.compile('\s{2,}')
    return regex_extra_space.sub(' ', regex_punctuation.sub(' ', line)).strip() 

def process_manifest(manifest_path):
    manifest_data = []
    with open(manifest_path, 'r', encoding='utf_8') as manifest_file:
        for line in manifest_file:
            data_entry = json.loads(line)
            #data_entry["audio_filepath"] = str(Path(data_entry['audio_filepath']).absolute())
            manifest_data.append(data_entry)
    return manifest_data


def sample_data(data, weight, update_whole_cache, p_cache):
    weight_factor = weight * p_cache if not update_whole_cache else weight
    sample_size = int(len(data) * weight_factor / torch.distributed.get_world_size()) 
    return random.sample(data, sample_size)
