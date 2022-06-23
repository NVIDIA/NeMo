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

import argparse
import json
import logging
import os
import random
import numpy as np
import copy 
import librosa as l
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

random.seed(42)
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    get_subsegments,
    get_embs_and_timestamps,
    get_uniqname_from_filepath,
    parse_scale_configs,
    perform_clustering,
    score_labels,
    segments_manifest_to_subsegments_manifest,
    write_rttm2manifest,
    rttm_to_labels,
    labels_to_pyannote_object


)

"""
This scipt converts a scp file where each line contains  
<absolute path of wav file> 
to a manifest json file. 
Args: 
--scp: scp file name
--id: index of speaker label in filename present in scp file that is separated by '/'
--out: output manifest file name
--split: True / False if you would want to split the  manifest file for training purposes
        you may not need this for test set. output file names is <out>_<train/dev>.json
        Defaults to False
--create_chunks: bool if you would want to chunk each manifest line to chunks of 3 sec or less
        you may not need this for test set, Defaults to False
"""

def rreplace(s, old, new):
    li = s.rsplit(old, 1)
    return new.join(li)

def write_file(name, lines, idx):
    with open(name, 'w') as fout:
        for i in idx:
            dic = lines[i]
            json.dump(dic, fout)
            fout.write('\n')
    logging.info("wrote", name)

def get_uniq_id_with_period(path):
    split_path = os.path.basename(path).split('.')[:-1]
    uniq_id = '.'.join(split_path) if len(split_path) > 1 else split_path[0]
    return uniq_id

def get_subsegment_dict(subsegments_manifest_file, window, shift, deci):
    _subsegment_dict = {}
    with open(subsegments_manifest_file, 'r') as subsegments_manifest:
        print(f"Reading subsegments_manifest_file: {subsegments_manifest_file}")
        segments = subsegments_manifest.readlines()
        for segment in segments:
            segment = segment.strip()
            dic = json.loads(segment)
            audio, offset, duration, label = dic['audio_filepath'], dic['offset'], dic['duration'], dic['label']
            subsegments = get_subsegments(offset=offset, window=window, shift=shift, duration=duration)
            uniq_id = get_uniq_id_with_period(audio)
            if uniq_id not in _subsegment_dict:
                _subsegment_dict[uniq_id] = {'ts' : [], 'json_dic': []}
            for subsegment in subsegments:
                start, dur = subsegment
            _subsegment_dict[uniq_id]['ts'].append([round(start, deci), round(start+dur, deci)])
            _subsegment_dict[uniq_id]['json_dic'].append(dic)
    return _subsegment_dict

def get_input_manifest_dict(input_manifest_path):
    input_manifest_dict = {}
    with open(input_manifest_path, 'r') as input_manifest_fp:
        json_lines = input_manifest_fp.readlines()
        for json_line in json_lines:
            dic = json.loads(json_line)
            dic["text"] = "-"
            uniq_id = get_uniqname_from_filepath(dic["audio_filepath"])
            input_manifest_dict[uniq_id] = dic
    return input_manifest_dict

# def write_truncated_subsegments(input_manifest_dict, _subsegment_dict, output_manifest_path, step_count, deci): 
    # with open(output_manifest_path, 'w') as output_manifest_fp:
        # for uniq_id, subseg_dict in _subsegment_dict.items():
            # print(f"Writing {uniq_id}")
            # subseg_array = np.array(subseg_dict['ts'])
            # subseg_array_idx = np.argsort(subseg_array, axis=0)
            # chunked_set_count = subseg_array_idx.shape[0] // step_count 

            # for idx in range(chunked_set_count-1):
                # chunk_index_stt = subseg_array_idx[:, 0][idx * step_count]
                # chunk_index_end = subseg_array_idx[:, 1][(idx+1)* step_count]
                # offset_sec = subseg_array[chunk_index_stt, 0]
                # end_sec = subseg_array[chunk_index_end, 1]
                # dur = round(end_sec - offset_sec, deci)
                # meta = input_manifest_dict[uniq_id]
                # import ipdb; ipdb.set_trace()

                # meta['offset'] = offset_sec
                # meta['duration'] = dur
                # json.dump(meta, output_manifest_fp)
                # output_manifest_fp.write("\n")

def write_truncated_subsegments(input_manifest_dict, _subsegment_dict, output_manifest_path, step_count, deci): 
    with open(output_manifest_path, 'w') as output_manifest_fp:
        for uniq_id, subseg_dict in _subsegment_dict.items():
            print(f"Writing {uniq_id}")
            subseg_array = np.array(subseg_dict['ts'])
            subseg_array_idx = np.argsort(subseg_array, axis=0)
            chunked_set_count = subseg_array_idx.shape[0] // step_count 

            for idx in range(chunked_set_count-1):
                chunk_index_stt = subseg_array_idx[:, 0][idx * step_count]
                chunk_index_end = subseg_array_idx[:, 1][(idx+1)* step_count]
                offset_sec = subseg_array[chunk_index_stt, 0]
                end_sec = subseg_array[chunk_index_end, 1]
                dur = round(end_sec - offset_sec, deci)
                meta = input_manifest_dict[uniq_id]
                meta['offset'] = offset_sec
                meta['duration'] = dur
                json.dump(meta, output_manifest_fp)
                output_manifest_fp.write("\n")
 
def main(input_manifest_path, output_manifest_path, window, shift, step_count, deci):
    if '.json' not in input_manifest_path:
        raise ValueError("input_manifest_path file should be .json file format")
    if output_manifest_path and '.json' not in output_manifest_path:
        raise ValueError("output_manifest_path file should be .json file format")
    elif not output_manifest_path:
        output_manifest_path = rreplace(input_manifest_path, '.json', f'_{step_count}seg.json')

    input_manifest_dict = get_input_manifest_dict(input_manifest_path)
    segment_manifest_path = rreplace(input_manifest_path, '.json', '_seg.json')
    subsegment_manifest_path = rreplace(input_manifest_path, '.json', '_subseg.json')
    min_subsegment_duration=0.05
    step_count = int(step_count)

    input_manifest_file = open(input_manifest_path, 'r').readlines()
    input_manifest_file = sorted(input_manifest_file)
    AUDIO_RTTM_MAP = audio_rttm_map(input_manifest_path)
    segments_manifest_file = write_rttm2manifest(AUDIO_RTTM_MAP, segment_manifest_path, deci)
    subsegments_manifest_file = subsegment_manifest_path
    segments_manifest_to_subsegments_manifest(
        segments_manifest_file,
        subsegments_manifest_file, 
        window,
        shift,
        min_subsegment_duration,
    )
    subsegments_dict = get_subsegment_dict(subsegments_manifest_file, window, shift, deci)
    write_truncated_subsegments(input_manifest_dict, subsegments_dict, output_manifest_path, step_count, deci)
    os.remove(segment_manifest_path)
    os.remove(subsegment_manifest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest_path", help="input json file name", type=str, required=True)
    parser.add_argument("--output_manifest_path", help="output manifest_file name", type=str, default=None, required=False)
    parser.add_argument("--window", help="Window length for segmentation", type=float, required=True)
    parser.add_argument("--shift", help="Shift length for segmentation", type=float, required=True)
    parser.add_argument("--deci", help="Rounding decimals", type=int, default=3, required=False)
    parser.add_argument(
        "--step_count",
        help="Number of the unit segments you want to create per utterance",
        required=True,
    )
    args = parser.parse_args()

    main(args.input_manifest_path, 
         args.output_manifest_path,
         args.window,
         args.shift,
         args.step_count,
         args.deci)
