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

import time
import copy
import csv
import itertools
import json
import os
import math
import copy
from collections import OrderedDict as od
from datetime import datetime
from itertools import permutations
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    get_uniqname_from_filepath,
    labels_to_rttmfile,
    rttm_to_labels,
    write_rttm2manifest,
)
from nemo.utils import logging

from nemo.collections.asr.parts.utils.decoder_timestamps_utils import FrameBatchASR_Logits, WERBPE_TS, ASR_TIMESTAMPS, WER_TS
from nemo.collections.asr.parts.utils.speaker_utils import get_contiguous_stamps, merge_stamps, labels_to_pyannote_object, rttm_to_labels, labels_to_rttmfile, get_uniqname_from_filepath, get_embs_and_timestamps, get_subsegments, isOverlap, getOverlapRange, getMergedRanges, getSubRangeList, fl2int, int2fl, combine_int_overlaps
import torch
from nemo.collections.asr.parts.utils.vad_utils import (
    generate_overlap_vad_seq,
    generate_vad_segment_table,
    get_vad_stream_status,
    prepare_manifest,
)
from nemo.collections.asr.models import OnlineDiarizer
from nemo.collections import nlp as nemo_nlp
import nemo.collections.asr as nemo_asr
from typing import Dict, List, Tuple, Type, Union
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from sklearn.preprocessing import OneHotEncoder
from nemo.collections.asr.parts.utils.streaming_utils import AudioFeatureIterator, FrameBatchASR
# , FrameBatchVAD
from omegaconf import OmegaConf


__all__ = ['word_error_rate', 'WER', 'move_dimension_to_the_front']

def score_labels(AUDIO_RTTM_MAP, all_reference, all_hypothesis, collar=0.25, ignore_overlap=True):
    """
    >>> [This function should be merged into speaker_utils.py]
    """
    metric = None
    if len(all_reference) == len(all_hypothesis):
        metric = DiarizationErrorRate(collar=2 * collar, skip_overlap=ignore_overlap)

        mapping_dict = {}
        for (reference, hypothesis) in zip(all_reference, all_hypothesis):
            ref_key, ref_labels = reference
            _, hyp_labels = hypothesis
            uem = AUDIO_RTTM_MAP[ref_key].get('uem_filepath', None)
            if uem is not None:
                uem = uem_timeline_from_file(uem_file=uem, uniq_name=ref_key)
            metric(ref_labels, hyp_labels, uem=uem, detailed=True)
            mapping_dict[ref_key] = metric.optimal_mapping(ref_labels, hyp_labels)

        DER = abs(metric)
        CER = metric['confusion'] / metric['total']
        FA = metric['false alarm'] / metric['total']
        MISS = metric['missed detection'] / metric['total']
        itemized_errors = (DER, CER, FA, MISS)

        logging.info(
            "Cumulative Results for collar {} sec and ignore_overlap {}: \n FA: {:.4f}\t MISS {:.4f}\t \
                Diarization ER: {:.4f}\t, Confusion ER:{:.4f}".format(
                collar, ignore_overlap, FA, MISS, DER, CER
            )
        )

        return metric, mapping_dict, itemized_errors
    else:
        logging.warning(
            "check if each ground truth RTTMs were present in provided manifest file. Skipping calculation of Diariazation Error Rate"
        )
        return None

def get_partial_ref_labels(pred_labels, ref_labels):
    last_pred_time = float(pred_labels[-1].split()[1])
    ref_labels_out = []
    for label in ref_labels:
        start, end, speaker = label.split()
        start, end = float(start), float(end)
        if last_pred_time <= start:
            pass
        elif start < last_pred_time <= end:
            label = f"{start} {last_pred_time} {speaker}"
            ref_labels_out.append(label)
        elif end < last_pred_time:
            ref_labels_out.append(label)
    return ref_labels_out

def get_online_DER_stats(DER, CER, FA, MISS, diar_eval_count, der_stat_dict, deci=3):
    der_dict = {
        "DER": round(100 * DER, deci),
        "CER": round(100 * CER, deci),
        "FA": round(100 * FA, deci),
        "MISS": round(100 * MISS, deci),
    }
    der_stat_dict['cum_DER'] += DER
    der_stat_dict['cum_CER'] += CER
    der_stat_dict['avg_DER'] = round(100 * der_stat_dict['cum_DER'] / diar_eval_count, deci)
    der_stat_dict['avg_CER'] = round(100 * der_stat_dict['cum_CER'] / diar_eval_count, deci)
    der_stat_dict['max_DER'] = round(max(der_dict['DER'], der_stat_dict['max_DER']), deci)
    der_stat_dict['max_CER'] = round(max(der_dict['CER'], der_stat_dict['max_CER']), deci)
    return der_dict, der_stat_dict

