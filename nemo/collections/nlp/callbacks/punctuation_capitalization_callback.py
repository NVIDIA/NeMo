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

import numpy as np
import torch

from nemo import logging
from nemo.collections.nlp.utils.callback_utils import get_classification_report, plot_confusion_matrix, tensor2list

__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']


def eval_iter_callback(tensors, global_vars):
    if "punct_all_preds" not in global_vars.keys():
        global_vars["punct_all_preds"] = []
    if "punct_all_labels" not in global_vars.keys():
        global_vars["punct_all_labels"] = []
    if "capit_all_preds" not in global_vars.keys():
        global_vars["capit_all_preds"] = []
    if "capit_all_labels" not in global_vars.keys():
        global_vars["capit_all_labels"] = []
    if "all_subtokens_mask" not in global_vars.keys():
        global_vars["all_subtokens_mask"] = []

    GLOBAL_KEYS = ['punct_labels', 'capit_labels', 'punct_preds', 'capit_preds']
    for key in GLOBAL_KEYS:
        if key not in global_vars:
            global_vars[key] = []

    output = {}
    for k, v in tensors.items():
        name = k.split('~~~')
        if len(name) > 1:
            output[name[0]] = torch.cat(v)

    subtokens_mask = output['subtokens_mask'] > 0.5
    global_vars['punct_preds'].extend(tensor2list(torch.argmax(output['punct_logits'], axis=-1)[subtokens_mask]))
    global_vars['capit_preds'].extend(tensor2list(torch.argmax(output['capit_logits'], axis=-1)[subtokens_mask]))
    global_vars['punct_labels'].extend(tensor2list(output['punct_labels'][subtokens_mask]))
    global_vars['capit_labels'].extend(tensor2list(output['capit_labels'][subtokens_mask]))


def eval_epochs_done_callback(global_vars, punct_label_ids, capit_label_ids, graph_fold=None, normalize_cm=True):
    '''
    Args:
      graph_fold (str): path to output folder
      normalize_cm (bool): flag to indicate whether to
        normalize confusion matrix
    '''
    results = {}
    punct_class_report = _eval_epochs_done_callback('punct', global_vars, punct_label_ids, graph_fold, normalize_cm)
    for label in punct_class_report:
        if label != 'accuracy':
            label_name = label[: label.index('(label id') - 1] if 'label id' in label else label
            results['pF1 ' + label_name] = round(punct_class_report[label]['f1-score'] * 100, 2)
            results['pPR ' + label_name] = round(punct_class_report[label]['precision'] * 100, 2)
            results['pR ' + label_name] = round(punct_class_report[label]['recall'] * 100, 2)

    capit_class_report = _eval_epochs_done_callback('capit', global_vars, capit_label_ids, graph_fold, normalize_cm)
    for label in capit_class_report:
        if label != 'accuracy':
            label_name = label[: label.index('(label id') - 1] if 'label id' in label else label
            results['cF1: ' + label_name] = round(capit_class_report[label]['f1-score'] * 100, 2)
            results['pPR ' + label_name] = round(capit_class_report[label]['precision'] * 100, 2)
            results['pR ' + label_name] = round(capit_class_report[label]['recall'] * 100, 2)

    logging.info(f'results: {results}')
    return results


def _eval_epochs_done_callback(task_name, global_vars, label_ids, graph_fold=None, normalize_cm=True):
    labels = np.array(global_vars[task_name + '_labels'])
    preds = np.array(global_vars[task_name + '_preds'])

    # calculate and plot confusion_matrix
    if graph_fold:
        plot_confusion_matrix(labels, preds, graph_fold, label_ids, normalize=normalize_cm, prefix=task_name)

    logging.info(f'{get_classification_report(labels, preds, label_ids)}')
    return get_classification_report(labels, preds, label_ids, output_dict=True)
