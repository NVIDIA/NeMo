import argparse
import os
import subprocess

import numpy as np
from kaldi_python_io import ArchiveWriter


def write_scp(root, filename, lines, train):
    assert len(lines) == len(train)
    filename=os.path.join(root,filename)
    with ArchiveWriter(filename + '.ark', filename + '.scp') as writer:
        for key, mat in zip(lines, train):
            writer.write(key, mat)
    print("wrote {}.ark".format(filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",help="embeddings root path", type=str, required=True)
    parser.add_argument("--train_embs", help="npy of train embs", type=str, required=True)
    parser.add_argument("--train_labels", help="npy of train labels", type=str, required=True)
    parser.add_argument("--eval_embs", help="npy of eval embb", type=str, required=True)
    parser.add_argument("--eval_labels", help="npy of eval labels", type=str, required=True)
    parser.add_argument("--stage", help="1 for test on already trained PLDA 2 otherwise", type=str, required=True)
    args = parser.parse_args()

    root = args.root

    if int(args.stage) < 2:
        train = np.load(args.train_embs)
        labels = np.load(args.train_labels)

        write_scp(root,'train', labels, train)

    eval = np.load(args.eval_embs)
    labels = np.load(args.eval_labels)

    write_scp(root,'dev', labels, eval)

    cmd = ['bash', 'train_plda.sh', root, args.stage]
    subprocess.run(cmd)
