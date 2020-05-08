# Copyright 2020 NVIDIA. All Rights Reserved.
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

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from tqdm import tqdm


def get_acc(data_root='./myExps/hi-mia/', emb='', emb_labels='', emb_size=512):
    basename = os.path.dirname(emb)
    X_test = np.load(emb)
    label_files = np.load(emb_labels)

    assert len(X_test) == len(label_files)
    trail_file = root + 'trials_1m'

    test_list = {}
    speaker_list = {}

    for idx, line in enumerate(label_files):
        line = line.strip()
        speaker = line.split('.')[0].split('_')[0]
        test_list[line] = idx

        if speaker in speaker_list:
            speaker_list[speaker].append(idx)
        else:
            speaker_list[speaker] = [idx]

    emb = int(emb_size)
    # import ipdb; ipdb.set_trace()
    tmp_file = open(trail_file, 'r').readlines()
    trail_score = open('trial_score.txt', 'w')

    trial_embs = []
    keys = []
    all_scores = []
    all_keys = []

    for line in tqdm(tmp_file):
        line = line.strip()
        x_speaker = line.split(' ')[0]
        y_speaker = line.split(' ')[1]

        X = np.zeros(emb,)
        for idx in speaker_list[x_speaker]:
            X = X + X_test[idx]

        X = X / len(speaker_list[x_speaker])

        if x_speaker not in keys:
            keys.append(x_speaker)
            trial_embs.extend([X])

        Y = np.zeros(emb,)
        for idx in speaker_list[y_speaker]:
            Y = Y + X_test[idx]

        Y = Y / len(speaker_list[y_speaker])

        if y_speaker not in keys:
            keys.append(y_speaker)
            trial_embs.extend([Y])

        # X=Y
        score = (X @ Y.T) / (((X @ X.T) * (Y @ Y.T)) ** 0.5)
        score = (score + 1) / 2

        all_scores.append(score)
        truth = 0 if line.split(' ')[-1] == 'nontarget' else 1

        all_keys.append(truth)

        trail_score.write(str(score) + "\t" + line.split(' ')[-1])
        trail_score.write('\n')

    np.save(basename + '/all_embs_himia.npy', np.asarray(trial_embs))
    np.save(basename + '/all_ids_himia.npy', np.asarray(keys))

    return np.asarray(all_scores), np.asarray(all_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", help="directory of embeddings location", type=str, required=True)
    parser.add_argument("--emb", help="embedding file name excluding npy type", type=str, required=True)
    parser.add_argument("--emb_labels", help="embedding file name excluding npy type", type=str, required=True)
    parser.add_argument("--emb_size", help="Embeddings size", type=int, required=True)
    args = parser.parse_args()
    root, emb, emb_labels, emb_size = args.data_root, args.emb, args.emb_labels, args.emb_size

    y_score, y = get_acc(data_root=root, emb=emb, emb_labels=emb_labels, emb_size=emb_size)
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    # cmd = ['./compute_scores.sh', task]
    # subprocess.run(cmd)
    print("EER: {:.2f}%".format(eer * 100))
