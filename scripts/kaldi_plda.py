import argparse
import os
import subprocess

import numpy as np
from kaldi_python_io import ArchiveWriter


def write_scp(filename, lines, train):
    assert len(lines) == len(train)
    if not os.path.exists('kaldi_files'):
        os.mkdir('kaldi_files')

    with ArchiveWriter('kaldi_files/' + filename + '.ark', 'kaldi_files/' + filename + '.scp') as writer:
        for key, mat in zip(lines, train):
            writer.write(key, mat)
    print("wrote {}.ark".format(filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_embs", help="npy of train embs", type=str)
    parser.add_argument("--train_labels", help="npy of train labels", type=str)
    parser.add_argument("--eval_embs", help="npy of eval embb", type=int)
    parser.add_argument("--eval_labels", help="npy of eval labels", type=str)
    parser.add_argument("--task", help="ffsvc task id", type=str)
    parser.add_argument("--stage", help="1 for test on already trained PLDA 2 otherwise", type=str)
    args = parser.parse_args()

    if args.stage < 2:
        train = np.load(args.train_embs)
        labels = np.load(args.train_labels)

        write_scp('train', labels, train)

    eval = np.load(args.eval_embs)
    labels = np.load(args.eval_labels)

    write_scp('dev', labels, eval)

    cmd = ['bash', 'train_plda.sh', 'kaldi_files/train.scp', 'kaldi_files/dev.scp', args.task, args.stage]
    subprocess.run(cmd)
