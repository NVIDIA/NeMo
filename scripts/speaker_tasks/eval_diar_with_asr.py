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

"""
This script evaluates multi-speaker ASR results (Diarization with ASR).

 python eval_diar_with_asr.py \

Args:
"""

import argparse
import json
import os
import random

import librosa as l
import numpy as np
import soundfile as sf
import sox
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.contrib.concurrent import process_map



from nemo.collections.asr.parts.utils.diarization_utils import (
convert_ctm_to_text,

)
from nemo.collections.asr.parts.utils.speaker_utils import (
get_uniqname_from_filepath,
rttm_to_labels,
labels_to_pyannote_object,
)

from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR as asr_diar_offline
from nemo.collections.asr.parts.utils.manifest_utils import read_file
from nemo.collections.asr.metrics.der import concat_perm_word_error_rate, score_labels
from nemo.collections.asr.metrics.wer import word_error_rate


def get_pyannote_objs_from_rttms(file_path_list):
    pyannote_obj_list = []
    for rttm_file in file_path_list: 
        rttm_file = rttm_file.strip()
        if rttm_file is not None and os.path.exists(rttm_file):
            uniq_id = get_uniqname_from_filepath(rttm_file)
            ref_labels = rttm_to_labels(rttm_file)
            reference = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
            pyannote_obj_list.append([uniq_id, reference])
    return pyannote_obj_list


def make_meta_dict(hyp_rttm_list, ref_rttm_list):
    meta_dict = {}
    for k, rttm_file in enumerate(ref_rttm_list):
        uniq_id = get_uniqname_from_filepath(rttm_file)
        meta_dict[uniq_id] = {"rttm_filepath":rttm_file.strip()}
        if hyp_rttm_list is not None:
            hyp_rttm_file = hyp_rttm_list[k]
            meta_dict[uniq_id].update({"hyp_rttm_filepath":hyp_rttm_file.strip()})
    return meta_dict

def make_trans_info_dict(hyp_json_list_path):
    trans_info_dict = {}
    for json_file in hyp_json_list_path:
        json_file = json_file.strip()
        with open(json_file) as jsf:
            json_data = json.load(jsf)
        uniq_id = get_uniqname_from_filepath(json_file)
        trans_info_dict[uniq_id] = json_data
    return trans_info_dict

def main(
    hyp_rttm_list_path,
    ref_rttm_list_path, 
    hyp_ctm_list_path, 
    ref_ctm_list_path, 
    hyp_json_list_path, 
    diar_eval_mode,
    root_path="./",
    ):

    hyp_rttm_list = read_file(hyp_rttm_list_path) if hyp_rttm_list_path else None
    ref_rttm_list = read_file(ref_rttm_list_path) if ref_rttm_list_path else None
    hyp_ctm_list = read_file(hyp_ctm_list_path) if hyp_ctm_list_path else None
    ref_ctm_list = read_file(ref_ctm_list_path) if ref_ctm_list_path else None
    hyp_json_list = read_file(hyp_json_list_path) if hyp_rttm_list_path else None

    meta_dict = make_meta_dict(hyp_rttm_list, ref_rttm_list)
    trans_info_dict = make_trans_info_dict(hyp_json_list)
    all_hypothesis = get_pyannote_objs_from_rttms(hyp_rttm_list)
    all_reference = get_pyannote_objs_from_rttms(ref_rttm_list)
    
    metric, mapping_dict, itemized_errors = score_labels(AUDIO_RTTM_MAP=meta_dict, 
                                                         all_reference=all_reference,
                                                         all_hypothesis=all_hypothesis, 
                                                         collar=0.25, 
                                                         ignore_overlap=False)

    if ref_ctm_list is not None:
        ref_ctm_list = [ x.strip() for x in ref_ctm_list]
        der_results = asr_diar_offline.gather_eval_results(metric=metric, 
                                                           mapping_dict=mapping_dict,
                                                           audio_rttm_map_dict=meta_dict,
                                                           trans_info_dict=trans_info_dict,
                                                           root_path=root_path)
        if hyp_ctm_list:
            hyp_ctm_list = [ x.strip() for x in hyp_ctm_list]
            wer_results = asr_diar_offline.evaluate(trans_info_dict=None,
                                                    audio_file_list=hyp_rttm_list,
                                                    ref_ctm_file_list=ref_ctm_list,
                                                    hyp_ctm_file_list=hyp_ctm_list,
                                                    )
        else:
            wer_results = asr_diar_offline.evaluate(trans_info_dict=trans_info_dict, 
                                                    audio_file_list=hyp_rttm_list,
                                                    ref_ctm_file_list=ref_ctm_list)

    asr_diar_offline.print_errors(der_results=der_results, 
                                  wer_results=wer_results)

    asr_diar_offline.write_session_level_result_in_csv(der_results=der_results, 
                                                       wer_results=wer_results, 
                                                       root_path=root_path,
                                                       csv_columns=asr_diar_offline.get_csv_columns())
    return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp_rttm_list", help="path to filelist of hypothesis RTTM files", type=str, required=True, default=None)
    parser.add_argument("--ref_rttm_list", help="path to filelist of reference RTTM files", type=str, required=True, default=None)
    parser.add_argument("--hyp_ctm_list", help="path to filelist of hypothesis CTM files", type=str, required=False, default=None)
    parser.add_argument("--ref_ctm_list", help="path to filelist of reference CTM files", type=str, required=False, default=None)
    parser.add_argument("--hyp_json_list", help="(Optional) path to filelist of hypothesis JSON files", type=str, required=False, default=None)
    parser.add_argument("--diar_eval_mode", help='evaluation mode: "all", "fair", "forgiving", "full"', type=str, required=False, default="all")
    parser.add_argument("--root_path", help='directory for saving result files', type=str, required=False, default="./")

    args = parser.parse_args()

    main(args.hyp_rttm_list,
         args.ref_rttm_list, 
         args.hyp_ctm_list, 
         args.ref_ctm_list, 
         args.hyp_json_list, 
         args.diar_eval_mode, 
         args.root_path, 
    )
