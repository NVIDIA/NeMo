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

import argparse
import json
from nemo.utils import logging
import os
import random
import numpy as np
import copy 
import librosa as l
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels 
import itertools

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
This scipt creates a manifest file for diarization training. If you specify `pairwise_rttm_output_folder`, the script generates
two-speaker subset of the original RTTM files. For example, an RTTM file with 4 speakers will obtain 6 different pairs and
6 RTTM files with two speakers in each RTTM file.

Args:
   --input_manifest_path: input json file name
   --output_manifest_path: output manifest_file name
   --pairwise_rttm_output_folder: Save two-speaker pair RTTM files
   --window: Window length for segmentation
   --shift: Shift length for segmentation
   --deci: Rounding decimals
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

def labels_to_rttmfile(labels, uniq_id, filename, out_rttm_dir):
    """
    Write rttm file with uniq_id name in out_rttm_dir with time_stamps in labels
    """
    filename = os.path.join(out_rttm_dir, filename + '.rttm')
    with open(filename, 'w') as f:
        for line in labels:
            line = line.strip()
            start, end, speaker = line.split()
            duration = float(end) - float(start)
            start = float(start)
            log = 'SPEAKER {} 1   {:.3f}   {:.3f} <NA> <NA> {} <NA> <NA>\n'.format(uniq_id, start, duration, speaker)
            f.write(log)

    return filename

def split_into_pairwise_rttm(audio_rttm_map, input_manifest_path, output_dir):
    """
    Create pairwise RTTM files and save it to `output_dir`. This function picks two speakers from the original RTTM files
    then saves the two-speaker subset of RTTM to `output_dir`.

    Args:
        audio_rttm_map (dict):
            A dictionary with keys of uniq id, which is being used to map audio files and corresponding rttm files
        input_manifest_path (str):
            Path of the input manifest file.
        output_dir (str):
            Path to the directory where the new RTTM files are saved.
    """
    input_manifest_dict = get_input_manifest_dict(input_manifest_path)
    rttmlist = []
    rttm_split_manifest_dict = {}
    split_audio_rttm_map = {}
    logging.info("Creating split RTTM files.")
    for uniq_id, line in tqdm(input_manifest_dict.items(), total=len(input_manifest_dict)):
        audiopath = line['audio_filepath']
        num_speakers = line['num_speakers']
        rttm_filepath = line['rttm_filepath']

        rttm = rttm_to_labels(rttm_filepath) 
        speakers = []
        j = 0
        while (len(speakers) < num_speakers):
            if rttm[j].split(' ')[2] not in speakers:
                speakers.append(rttm[j].split(' ')[2])
            j += 1
        base_fn = audiopath.split('/')[-1].replace('.wav','')
        for pair in itertools.combinations(speakers, 2):
            i, target_rttm = 0, []
            while (i < len(rttm)):
                entry = rttm[i]
                sp_id = entry.split(' ')[2]
                if sp_id in pair:
                    target_rttm.append(entry)
                i += 1
           
            pair_string = f"_{pair[0]}_{pair[1]}" 
            uniq_id_pair = uniq_id + pair_string
            filename = base_fn + pair_string
            labels_to_rttmfile(target_rttm, base_fn, filename, output_dir)
            rttm_path = output_dir + filename + ".rttm"
            rttmlist.append(rttm_path)
            line_mod = copy.deepcopy(line)
            line_mod['rttm_filepath'] = rttm_path
            meta = copy.deepcopy(audio_rttm_map[uniq_id])
            meta['rttm_filepath'] = rttm_path
            rttm_split_manifest_dict[uniq_id_pair] = line_mod
            split_audio_rttm_map[uniq_id_pair] = meta

    with open(os.path.join(output_dir, "ami_split_rttm.list"), "w") as f:
        for rttm in rttmlist:
            f.write(rttm + "\n")
    return rttm_split_manifest_dict, split_audio_rttm_map

def get_subsegment_dict(subsegments_manifest_file, window, shift, deci):
    _subsegment_dict = {}
    with open(subsegments_manifest_file, 'r') as subsegments_manifest:
        logging.info(f"Reading subsegments_manifest_file: {subsegments_manifest_file}")
        segments = subsegments_manifest.readlines()
        for segment in tqdm(segments, total=len(segments)):
            segment = segment.strip()
            dic = json.loads(segment)
            audio, offset, duration, label = dic['audio_filepath'], dic['offset'], dic['duration'], dic['label']
            subsegments = get_subsegments(offset=offset, window=window, shift=shift, duration=duration)
            if dic['uniq_id'] is not None:
                uniq_id = dic['uniq_id']
            else:
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

def write_truncated_subsegments(input_manifest_dict, _subsegment_dict, output_manifest_path, step_count, deci): 
    with open(output_manifest_path, 'w') as output_manifest_fp:
        logging.info("Writing truncated subsegments.")
        for uniq_id, subseg_dict in tqdm(_subsegment_dict.items(), total=len(_subsegment_dict)):
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
 
def main(input_manifest_path, output_manifest_path, pairwise_rttm_output_folder, window, shift, step_count, deci):
    
    if '.json' not in input_manifest_path:
        raise ValueError("input_manifest_path file should be .json file format")
    if output_manifest_path and '.json' not in output_manifest_path:
        raise ValueError("output_manifest_path file should be .json file format")
    elif not output_manifest_path:
        output_manifest_path = rreplace(input_manifest_path, '.json', f'.{step_count}seg.json')
    
    if pairwise_rttm_output_folder is not None:
        if not pairwise_rttm_output_folder.endswith('/'):
            pairwise_rttm_output_folder = f"{pairwise_rttm_output_folder}/"
        org_audio_rttm_map = audio_rttm_map(input_manifest_path)
        input_manifest_dict, AUDIO_RTTM_MAP = split_into_pairwise_rttm(audio_rttm_map=org_audio_rttm_map, input_manifest_path=input_manifest_path, output_dir=pairwise_rttm_output_folder)
    else:
        input_manifest_dict = get_input_manifest_dict(input_manifest_path)
        AUDIO_RTTM_MAP = audio_rttm_map(input_manifest_path)
    
    segment_manifest_path = rreplace(input_manifest_path, '.json', '_seg.json')
    subsegment_manifest_path = rreplace(input_manifest_path, '.json', '_subseg.json')
    min_subsegment_duration=0.05
    step_count = int(step_count)

    input_manifest_file = open(input_manifest_path, 'r').readlines()
    input_manifest_file = sorted(input_manifest_file)
    segments_manifest_file = write_rttm2manifest(AUDIO_RTTM_MAP, segment_manifest_path, deci)
    subsegments_manifest_file = subsegment_manifest_path
    
    logging.info("Creating subsegments.")
    segments_manifest_to_subsegments_manifest(
        segments_manifest_file,
        subsegments_manifest_file, 
        window,
        shift,
        min_subsegment_duration,
        include_uniq_id=True,
    )
    subsegments_dict = get_subsegment_dict(subsegments_manifest_file, window, shift, deci)
    write_truncated_subsegments(input_manifest_dict, subsegments_dict, output_manifest_path, step_count, deci)
    os.remove(segment_manifest_path)
    os.remove(subsegment_manifest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest_path", help="input json file name", type=str, required=True)
    parser.add_argument("--output_manifest_path", help="output manifest_file name", type=str, default=None, required=False)
    parser.add_argument("--pairwise_rttm_output_folder", help="Save two-speaker pair RTTM files", type=str, default=None, required=False)
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
         args.pairwise_rttm_output_folder,
         args.window,
         args.shift,
         args.step_count,
         args.deci)

