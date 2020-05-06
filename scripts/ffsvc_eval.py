import argparse
import os
import re

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from tqdm import tqdm


def get_acc(root='./myExps/ffsvc/', name='', emb=512, task='1'):
    # name = name + '_emb_' + str(int(emb))
    basename = os.path.join(root, name)
    X_test = np.load(basename + '.npy')
    label_files = np.load(basename + '_labels.npy')

    print(X_test.shape)
    trail_file = os.path.join(root, 'task' + task + '.trials.kays')
    print(trail_file)
    test_list = {}

    assert len(label_files) == len(X_test)

    sub_list = {}
    for idx in range(len(label_files)):
        line = label_files[idx].strip()
        wav_id = line.split('/')[-1].split()[0][:-4]
        test_list[wav_id] = idx

        record = wav_id.split('_')[:2]
        record = '_'.join(record)

        if record in sub_list:
            sub_list[record].append(wav_id)
        else:
            sub_list[record] = [wav_id]

    # import ipdb; ipdb.set_trace()
    tmp_file = open(trail_file, 'r').readlines()
    trail_score = open('trial_score.txt', 'w')

    trial_embs = []
    keys = []
    all_scores = []
    all_keys = []
    for line in tqdm(tmp_file):
        line = line.strip()
        X_idx = test_list[line.split()[0][:-4]]
        exp = line.split()[1][:-4]
        exp = exp.replace('{}', '.*')
        r = re.compile(exp)
        values = sub_list['_'.join(exp.split('_')[:2])]
        wav_ids = list(filter(r.match, values))  # [-1]
        # import ipdb; ipdb.set_trace()

        X = X_test[X_idx]

        Y = np.zeros(int(emb),)
        t_score = []
        for wav_id in wav_ids:
            Y_idx = test_list[wav_id]
            Y = Y + X_test[Y_idx]

            # if wav_id+'.wav' not in keys:
            #     trial_embs.extend([X_test[Y_idx]])
            #     keys.append(wav_id+'.wav')

        Y = Y / len(wav_ids)

        if line.split()[0] not in keys:
            trial_embs.extend([X])
            keys.append(line.split()[0])

        t = line.split()[1].replace('{}', 'average')

        if t not in keys:
            trial_embs.extend([Y])
            keys.append(t)

        score = (X @ Y.T) / (((X @ X.T) * (Y @ Y.T)) ** 0.5)
        score = (score + 1) / 2
        all_scores.append(score)
        truth = 0 if line.split(' ')[-1] == 'nontarget' else 1

        all_keys.append(truth)
        # score = max(t_score) #/len(t_score)
        trail_score.write(str(score) + "\t" + line.split(' ')[-1])
        trail_score.write('\n')
        # new_file.write(line.split()[0]+' '+wav_ids[1]+'.wav '+line.split(' ')[-1])
        # new_file.write('\n')

    np.save('all_embs_task' + task + '.npy', np.asarray(trial_embs))
    np.save('all_ids_task' + task + '.npy', np.asarray(keys))

    return np.asarray(all_scores), np.asarray(all_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="directory of embeddings location", type=str)
    parser.add_argument("--emb_name", help="embedding file name excluding npy type", type=str)
    parser.add_argument("--emb_size", help="Embeddings size", type=int)
    parser.add_argument("--task", help="FFSVC task number", type=str)
    args = parser.parse_args()
    root, name, emb, task = args.root, args.emb_name, args.emb_size, args.task
    y_score, y = get_acc(root=root, name=name, emb=emb, task=task)
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    # cmd = ['./compute_scores.sh', task]
    # subprocess.run(cmd)
    print("EER %:{:.3f}".format(eer))
