# =============================================================================
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
# =============================================================================

import random

import numpy as np

from nemo import logging
from nemo.collections.nlp.utils.callback_utils import (
    get_classification_report,
    get_f1_scores,
    list2str,
    plot_confusion_matrix,
    tensor2list,
)

__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']


def eval_iter_callback(tensors, global_vars):
    if "all_preds" not in global_vars.keys():
        global_vars["all_preds"] = []
    if "all_labels" not in global_vars.keys():
        global_vars["all_labels"] = []
    if "all_subtokens_mask" not in global_vars.keys():
        global_vars["all_subtokens_mask"] = []

    all_subtokens_mask, all_logits, all_labels = [], [], []

    for kv, v in tensors.items():
        if kv.startswith('logits'):
            for v_tensor in v:
                for logit_tensor in v_tensor:
                    all_logits.append(tensor2list(logit_tensor))

        elif kv.startswith('labels'):
            for v_tensor in v:
                for label_tensor in v_tensor:
                    all_labels.extend(tensor2list(label_tensor))

        elif kv.startswith('subtokens_mask'):
            for v_tensor in v:
                for subtokens_mask_tensor in v_tensor:
                    all_subtokens_mask.extend(tensor2list(subtokens_mask_tensor))

    all_preds = list(np.argmax(np.asarray(all_logits), 2).flatten())
    global_vars["all_preds"].extend(all_preds)
    global_vars["all_labels"].extend(all_labels)
    global_vars["all_subtokens_mask"].extend(all_subtokens_mask)


def eval_epochs_done_callback(global_vars, label_ids, graph_fold=None, normalize_cm=True):
    labels = np.asarray(global_vars['all_labels'])
    preds = np.asarray(global_vars['all_preds'])
    subtokens_mask = np.asarray(global_vars['all_subtokens_mask']) > 0.5

    labels = labels[subtokens_mask]
    preds = preds[subtokens_mask]

    # print predictions and labels for a small random subset of data
    sample_size = 20
    i = 0
    if preds.shape[0] > sample_size + 1:
        i = random.randint(0, preds.shape[0] - sample_size - 1)
    logging.info("Sampled preds: [%s]" % list2str(preds[i : i + sample_size]))
    logging.info("Sampled labels: [%s]" % list2str(labels[i : i + sample_size]))

    accuracy = sum(labels == preds) / labels.shape[0]
    logging.info(f'Accuracy: {accuracy}')

    f1_scores = get_f1_scores(labels, preds, average_modes=['weighted', 'macro', 'micro'])
    for k, v in f1_scores.items():
        logging.info(f'{k}: {v}')

    classification_report = get_classification_report(labels, preds, label_ids)
    logging.info(classification_report)

    # calculate and plot confusion_matrix
    if graph_fold:
        plot_confusion_matrix(labels, preds, graph_fold, label_ids, normalize=normalize_cm)

    return dict({'Accuracy': accuracy})
