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
from sklearn.metrics import classification_report

from nemo import logging
from nemo.collections.nlp.utils.callback_utils import list2str, plot_confusion_matrix, tensor2list

__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']


def eval_iter_callback(tensors, global_vars, eval_data_layer):
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
                    logits_lists.append(tensor2list(logit_tensor))

        if 'labels' in kv:
            for v_tensor in v:
                for label_tensor in v_tensor:
                    labels_lists.append(tensor2list(label_tensor))

    preds = list(np.argmax(np.asarray(logits_lists), 1))
    global_vars["all_preds"].extend(preds)
    global_vars["all_labels"].extend(labels_lists)


def eval_epochs_done_callback(global_vars, graph_fold):
    labels = np.asarray(global_vars['all_labels'])
    preds = np.asarray(global_vars['all_preds'])
    accuracy = sum(labels == preds) / labels.shape[0]
    logging.info(f'Accuracy: {accuracy}')

    # print predictions and labels for a small random subset of data
    sample_size = 20
    i = 0
    if preds.shape[0] > sample_size + 1:
        i = random.randint(0, preds.shape[0] - sample_size - 1)
    logging.info("Sampled preds: [%s]" % list2str(preds[i : i + sample_size]))
    logging.info("Sampled labels: [%s]" % list2str(labels[i : i + sample_size]))
    plot_confusion_matrix(labels, preds, graph_fold)
    logging.info(classification_report(labels, preds, digits=4))
    return dict({"accuracy": accuracy})
