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
import subprocess

import numpy as np
from kaldi_python_io import ArchiveWriter


def write_scp(root, filename, lines, train):
    assert len(lines) == len(train)
    filename = os.path.join(root, filename)
    with ArchiveWriter(filename + '.ark', filename + '.scp') as writer:
        for key, mat in zip(lines, train):
            writer.write(key, mat)
    print("wrote {}.ark".format(filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="embeddings root path", type=str, required=True)
    parser.add_argument("--train_embs", help="npy of train embs for PLDA training", type=str, required=True)
    parser.add_argument("--train_labels", help="npy of train labels for PLDA training", type=str, required=True)
    parser.add_argument("--eval_embs", help="npy of eval embb for PLDA testing", type=str, required=True)
    parser.add_argument("--eval_labels", help="npy of eval labels for PLDA testing", type=str, required=True)
    parser.add_argument("--stage", help="1 for test on already trained PLDA 2 otherwise", type=str, required=True)
    args = parser.parse_args()

    root = args.root

    if int(args.stage) < 2:
        train = np.load(args.train_embs)
        labels = np.load(args.train_labels)

        write_scp(root, 'train', labels, train)

    eval = np.load(args.eval_embs)
    labels = np.load(args.eval_labels)

    write_scp(root, 'dev', labels, eval)

    cmd = ['bash', 'train_plda.sh', root, args.stage]
    subprocess.run(cmd)
