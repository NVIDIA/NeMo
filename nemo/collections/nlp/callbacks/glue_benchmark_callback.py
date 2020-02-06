# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
Utility functions for GLUE tasks
Some transformer of this code were adapted from the HuggingFace library at
https://github.com/huggingface/transformers
"""
import os
import random

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef

from nemo import logging
from nemo.collections.nlp.utils.callback_utils import list2str, tensor2list

__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']


def eval_iter_callback(tensors, global_vars):
    if "all_preds" not in global_vars.keys():
        global_vars["all_preds"] = []
    if "all_labels" not in global_vars.keys():
        global_vars["all_labels"] = []

    logits_lists = []
    preds_lists = []
    labels_lists = []

    for kv, v in tensors.items():
        # for GLUE classification tasks
        if 'logits' in kv:
            for v_tensor in v:
                for logit_tensor in v_tensor:
                    logits_lists.append(tensor2list(logit_tensor))
        # for GLUE STS-B task (regression)
        elif 'preds' in kv:
            for v_tensor in v:
                for pred_tensor in v_tensor:
                    preds_lists.append(tensor2list(pred_tensor))
        if 'labels' in kv:
            for v_tensor in v:
                for label_tensor in v_tensor:
                    labels_lists.append(tensor2list(label_tensor))

    if len(logits_lists) > 0:
        preds = list(np.argmax(np.asarray(logits_lists), 1))
    elif len(preds_lists) > 0:
        preds = list(np.squeeze(np.asarray(preds_lists)))

    global_vars["all_preds"].extend(preds)
    global_vars["all_labels"].extend(labels_lists)


def eval_epochs_done_callback(global_vars, output_dir, task_name):
    labels = np.asarray(global_vars['all_labels'])
    preds = np.asarray(global_vars['all_preds'])

    # print predictions and labels for a small random subset of data
    sample_size = 20
    i = 0
    if preds.shape[0] > sample_size + 1:
        i = random.randint(0, preds.shape[0] - sample_size - 1)

    logging.info("Task name: %s" % task_name.upper())
    logging.info("Sampled preds: [%s]" % list2str(preds[i : i + sample_size]))
    logging.info("Sampled labels: [%s]" % list2str(labels[i : i + sample_size]))

    results = compute_metrics(task_name, preds, labels)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, task_name + '.txt'), 'w') as f:
        f.write('labels\t' + list2str(labels) + '\n')
        f.write('preds\t' + list2str(preds) + '\n')

    logging.info(results)

    return results


def accuracy(preds, labels):
    return {"acc": (preds == labels).mean()}


def acc_and_f1(preds, labels):
    accuracy = (preds == labels).mean()
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {"acc": accuracy, "f1": f1, "acc_and_f1": (accuracy + f1) / 2}


def mcc(preds, labels):
    return {"mcc": matthews_corrcoef(labels, preds)}


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {"pearson": pearson_corr, "spearmanr": spearman_corr, "corr": (pearson_corr + spearman_corr) / 2}


def compute_metrics(task_name, preds, labels):
    if len(preds) != len(labels):
        raise ValueError("Predictions and labels must have the same lenght")

    metric_fn = accuracy
    if task_name == 'cola':
        metric_fn = mcc
    elif task_name in ['mrpc', 'qqp']:
        metric_fn = acc_and_f1
    elif task_name == 'sts-b':
        metric_fn = pearson_and_spearman

    return metric_fn(preds, labels)
