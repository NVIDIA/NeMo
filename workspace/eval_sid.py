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
import os
import pickle as pkl
import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

"""
This script faciliates to get EER % based on cosine-smilarity 
for Voxceleb dataset.

Args:
    trial_file str: path to voxceleb trial file
    emb : path to pickle file of embeddings dictionary (generated from spkr_get_emb.py)
    save_kaldi_emb: if required pass this argument to save kaldi embeddings for KALDI PLDA training later
    Note: order of audio files in manifest file should match the embeddings
"""


def get_acc(trial_file='', emb='', save_kaldi_emb=False):

    dirname = os.path.dirname(trial_file)
    with open(emb, 'rb') as f:
        emb = pkl.load(f)
    trial_embs = []
    keys = []
    all_scores = []
    all_keys = []

    # for each trials in trial file
    with open(trial_file, 'r') as f:
        tmp_file = f.readlines()
        for line in tqdm(tmp_file):
            line = line.strip()
            truth, x_speaker, y_speaker = line.split()

            x_speaker = x_speaker.split('/')
            x_speaker_tag = '@'.join(x_speaker)

            y_speaker = y_speaker.split('/')
            y_speaker_tag = '@'.join(y_speaker)

            if x_speaker_tag not in emb:
                x_speaker_tag = '@'.join(x_speaker[-3:])
            if y_speaker_tag not in emb:
                y_speaker_tag = '@'.join(y_speaker[-3:])

            if x_speaker_tag not in emb or y_speaker_tag not in emb:
                continue

            X = emb[x_speaker_tag]
            Y = emb[y_speaker_tag]

            if save_kaldi_emb and x_speaker not in keys:
                keys.append(x_speaker)
                trial_embs.extend([X])

            if save_kaldi_emb and y_speaker not in keys:
                keys.append(y_speaker)
                trial_embs.extend([Y])

            score = np.dot(X, Y) / ((np.dot(X, X) * np.dot(Y, Y)) ** 0.5)
            score = (score + 1) / 2

            all_scores.append(score)
            try:
                truth = int(truth)
            except:
                print(line)
                import ipdb

                ipdb.set_trace()
            all_keys.append(truth)

    return np.asarray(all_scores), np.asarray(all_keys)


def compute_MinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_file", help="path to voxceleb trial file", type=str, required=True)
    parser.add_argument("--emb", help="path to numpy file of embeddings", type=str, required=True)
    parser.add_argument(
        "--save_kaldi_emb",
        help=":save kaldi embeddings for KALDI PLDA training later",
        required=False,
        action='store_true',
    )

    args = parser.parse_args()
    trial_file, emb, save_kaldi_emb = args.trial_file, args.emb, args.save_kaldi_emb

    y_score, y = get_acc(trial_file=trial_file, emb=emb, save_kaldi_emb=save_kaldi_emb)
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    auroc = roc_auc_score(y_true=y, y_score=y_score)

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    min_dcf, min_c_det_threshold = compute_MinDcf(
        fnrs=fnr, fprs=fpr, thresholds=thresholds, p_target=0.01, c_miss=1, c_fa=1
    )

    print(
        f"EER: {eer*100:.2f} w/ threshold {eer_threshold:.4f}, MinDCF: {min_dcf:.2f} w/ threshold {min_c_det_threshold:.4f}, AUROC: {auroc*100:.2f}"
    )
