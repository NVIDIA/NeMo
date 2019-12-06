# Copyright (c) 2019 NVIDIA Corporation
__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']

import os
import random
import time

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from nemo_nlp.data.datasets.utils import list2str, tensor2list
from nemo.utils.exp_logging import get_logger


logger = get_logger('')


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

    all_subtokens_mask = []
    punct_all_logits, punct_all_labels = [], []
    capit_all_logits, capit_all_labels = [], []

    for kv, v in tensors.items():
        if 'Punctuation' in kv and 'logits' in kv:
            for v_tensor in v:
                for logit_tensor in v_tensor:
                    punct_all_logits.append(tensor2list(logit_tensor))

        elif kv.startswith('punct_labels'):
            for v_tensor in v:
                for label_tensor in v_tensor:
                    punct_all_labels.extend(tensor2list(label_tensor))

        elif 'Capitalization' in kv and 'logits' in kv:
            for v_tensor in v:
                for logit_tensor in v_tensor:
                    capit_all_logits.append(tensor2list(logit_tensor))

        elif kv.startswith('capit_labels'):
            for v_tensor in v:
                for label_tensor in v_tensor:
                    capit_all_labels.extend(tensor2list(label_tensor))

        elif kv.startswith('subtokens_mask'):
            for v_tensor in v:
                for subtokens_mask_tensor in v_tensor:
                    all_subtokens_mask.extend(
                        tensor2list(subtokens_mask_tensor))

    punct_all_preds = list(np.argmax(np.asarray(punct_all_logits), 2).flatten())
    global_vars["punct_all_preds"].extend(punct_all_preds)
    global_vars["punct_all_labels"].extend(punct_all_labels)

    capit_all_preds = list(np.argmax(np.asarray(capit_all_logits), 2).flatten())
    global_vars["capit_all_preds"].extend(capit_all_preds)
    global_vars["capit_all_labels"].extend(capit_all_labels)

    global_vars["all_subtokens_mask"].extend(all_subtokens_mask)


def eval_epochs_done_callback(global_vars,
                              punct_label_ids,
                              capit_label_ids,
                              graph_fold=None,
                              none_label_id=0):

    punct_accuracy = _eval_epochs_done_callback('punct',
                                       global_vars,
                                       punct_label_ids,
                                       graph_fold=None,
                                       none_label_id=0)

    capit_accuracy = _eval_epochs_done_callback('capit',
                                       global_vars,
                                       capit_label_ids,
                                       graph_fold=None,
                                       none_label_id=0)

    return {"Punctuation task accuracy": punct_accuracy,
            "Capitalization task accuracy": capit_accuracy}


def _eval_epochs_done_callback(task_name,
                               global_vars,
                               label_ids,
                               graph_fold=None,
                               none_label_id=0):
    labels = np.asarray(global_vars[task_name + '_all_labels'])
    preds = np.asarray(global_vars[task_name + '_all_preds'])
    subtokens_mask = np.asarray(global_vars['all_subtokens_mask']) > 0.5

    labels = labels[subtokens_mask]
    preds = preds[subtokens_mask]

    accuracy = sum(labels == preds) / labels.shape[0]
    logger.info(f'Accuracy for task {task_name}: {accuracy}')

    # print predictions and labels for a small random subset of data
    sample_size = 20
    i = 0
    if preds.shape[0] > sample_size + 1:
        i = random.randint(0, preds.shape[0] - sample_size - 1)
    logger.info("Sampled preds: [%s]" % list2str(preds[i:i+sample_size]))
    logger.info("Sampled labels: [%s]" % list2str(labels[i:i+sample_size]))

    # remove labels from label_ids that don't appear in the dev set
    used_labels = set(labels) | set(preds)
    label_ids = \
        {k: label_ids[k] for k, v in label_ids.items() if v in used_labels}

    logger.info(classification_report(labels, preds, target_names=label_ids))

    # calculate and plot confusion_matrix
    if graph_fold:
        ids_to_labels = {label_ids[k]: k for k in label_ids}
        classes = [0] + \
            [ids_to_labels[id] for id in sorted(label_ids.values())]

        cm = confusion_matrix(labels, preds)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title='Confusion matrix for task {task_name}',
               ylabel='True label',
               xlabel='Predicted label')
        cax = ax.matshow(cm)
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        os.makedirs(graph_fold, exist_ok=True)
        plt.savefig(os.path.join(graph_fold, time.strftime('%Y%m%d-%H%M%S')))

    return accuracy

