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

from nemo.collections.asr.parts.vad_utils import vad_tune_threshold_on_dev
from nemo.utils import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold_range", help="range of threshold in list 'START,END,STEP' to be tuned on", required=True
    )
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

    try:
        start, stop, step = [float(i) for i in args.threshold_range.split(",")]
        thresholds = np.arange(start, stop, step)
    except:
        raise ValueError("Theshold input is invalid! Please enter it as a 'START,STOP,STEP' ")

    best_threhsold = vad_tune_threshold_on_dev(
        thresholds, args.vad_pred, args.groundtruth_RTTM, args.vad_pred_method, args.focus_metric
    )
    logging.info(f"Best threshold selected from {thresholds} is {best_threhsold}!")
