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

import numpy as np

from nemo.collections.asr.parts.utils.vad_utils import vad_tune_threshold_on_dev
from nemo.utils import logging

"""
This script is designed for thresholds tuning for postprocessing of VAD
See details about it in nemo/collections/asr/parts/utils/binarization

Usage:
python vad_tune_threshold.py --onset_range="0,1,0.2" --offset_range="0,1,0.2" --min_duration_on_range="0.1,0.8,0.05" --min_duration_off_range="0.1,0.8,0.05" \
--vad_pred=<FULL PATH OF FOLDER OF FRAME LEVEL PREDICTION FILES> \
--groundtruth_RTTM=<DIRECTORY OF VAD PREDICTIONS OR A FILE CONTAINS THE PATHS OF THEM> \
--vad_pred_method="median"

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onset_range", help="range of onset in list 'START,END,STEP' to be tuned on")
    parser.add_argument("--offset_range", help="range of offset in list 'START,END,STEP' to be tuned on")
    parser.add_argument("--min_duration_on_range", help="range of min_duration_on in list 'START,END,STEP' to be tuned on")
    parser.add_argument("--min_duration_off_range", help="range of min_duration_off in list 'START,END,STEP' to be tuned on")

    parser.add_argument(
        "--vad_pred", help="Directory of vad predictions or a file contains the paths of them.", required=True
    )
    parser.add_argument(
        "--groundtruth_RTTM",
        help="Directory of groundtruch rttm files or a file contains the paths of them",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--vad_pred_method",
        help="suffix of prediction file. Should be either in 'frame', 'mean' or 'median'",
        required=True,
    )
    parser.add_argument(
        "--focus_metric",
        help="metrics we care most when tuning threshold. Should be either in 'DetER', 'FA', 'MISS' ",
        type=str,
        default='DetER',
    )
    args = parser.parse_args()

    params = {}
    try:
        # if not input range for values of parameters, use default value defined in function binarization in nemo/collections/asr/parts/utils/vad_utils.py
        if args.onset_range:
            start, stop, step = [float(i) for i in args.onset_range.split(",")]
            onset = np.arange(start, stop, step)
            params['onset'] = onset

        if args.offset_range:
            start, stop, step = [float(i) for i in args.offset_range.split(",")]
            offset = np.arange(start, stop, step)
            params['offset'] = onset

        if args.min_duration_on_range:
            start, stop, step = [float(i) for i in args.min_duration_on_range.split(",")]
            min_duration_on = np.arange(start, stop, step)
            params['min_duration_on'] = min_duration_on

        if args.min_duration_off_range:
            start, stop, step = [float(i) for i in args.min_duration_off_range.split(",")]
            min_duration_off = np.arange(start, stop, step)
            params['min_duration_off'] = min_duration_off

    except:
        raise ValueError("Theshold input is invalid! Please enter it as a 'START,STOP,STEP' ")

    best_threhsold, optimal_scores = vad_tune_threshold_on_dev(
        params, args.vad_pred, args.groundtruth_RTTM, args.vad_pred_method, args.focus_metric
    )
    logging.info(f"Best combination of thresholds for binarization selected from input ranges is {best_threhsold}, and the optimal score is {optimal_scores}")
