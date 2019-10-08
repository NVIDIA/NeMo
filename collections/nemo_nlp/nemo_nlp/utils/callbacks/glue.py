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
__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']

import os
import random 
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from nemo.utils.exp_logging import get_logger

logger = get_logger('')


def eval_iter_callback(tensors, global_vars, eval_data_layer, output_mode):
    if "all_preds" not in global_vars.keys():
        global_vars["all_preds"] = []
    if "all_labels" not in global_vars.keys():
        global_vars["all_labels"] = []

    logits_lists = []
    labels_lists = []

    for kv, v in tensors.items():
     
        if 'logits' in kv:
            for v_tensor in v:
                for logit_tensor in v_tensor:
                    logits_lists.append(logit_tensor.detach().cpu().tolist())
        
        if 'labels' in kv:
            for v_tensor in v:
                for label_tensor in v_tensor:
                    labels_lists.append(label_tensor.detach().cpu().tolist())

    if output_mode == 'classification':
        preds = list(np.argmax(np.asarray(logits_lists), 1))
    elif output_mode == 'regression':
        preds = list(np.squeeze(np.asarray(logits_lists)))
    global_vars["all_preds"].extend(preds)
    global_vars["all_labels"].extend(labels_lists)

def list2str(l):
    return ' '.join([str(j) for j in l])

def eval_epochs_done_callback(global_vars, output_dir, task_name):
    labels = np.asarray(global_vars['all_labels'])
    preds = np.asarray(global_vars['all_preds'])

    i = 0
    if preds.shape[0] > 21:
        i = random.randint(0, preds.shape[0] - 21)
        
    logger.info("Task name: %s" % task_name.upper())
    logger.info("Sampled preds: [%s]" % list2str(preds[i:i+20]))
    logger.info("Sampled labels: [%s]" % list2str(labels[i:i+20]))
    
    results = compute_metrics(task_name, preds, labels)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, task_name + '.txt'), 'w') as f:
        f.write('labels\t' + list2str(labels) + '\n')
        f.write('preds\t' + list2str(preds) + '\n')
    
    logger.info(results)

    return results

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)
