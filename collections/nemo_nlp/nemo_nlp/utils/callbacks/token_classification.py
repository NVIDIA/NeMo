# Copyright (c) 2019 NVIDIA Corporation
__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']

import os
import random
import time

import numpy as np
from sklearn.metrics import classification_report

from nemo.utils.exp_logging import get_logger


logger = get_logger('')


def tensor2list(tensor):
    return tensor.detach().cpu().tolist()


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

        if kv.startswith('labels'):
            for v_tensor in v:
                for label_tensor in v_tensor:
                    all_labels.extend(tensor2list(label_tensor))

        if kv.startswith('subtokens_mask'):
            for v_tensor in v:
                for subtokens_mask_tensor in v_tensor:
                    all_subtokens_mask.extend(
                        tensor2list(subtokens_mask_tensor))

    all_preds = list(np.argmax(np.asarray(all_logits), 2).flatten())
    global_vars["all_preds"].extend(all_preds)
    global_vars["all_labels"].extend(all_labels)
    global_vars["all_subtokens_mask"].extend(all_subtokens_mask)


def list2str(l):
    return ' '.join([str(j) for j in l])


def eval_epochs_done_callback(global_vars, label_ids, none_label_id=0):
    labels = np.asarray(global_vars['all_labels'])
    preds = np.asarray(global_vars['all_preds'])
    subtokens_mask = np.asarray(global_vars['all_subtokens_mask'])

    labels = labels[subtokens_mask]
    preds = preds[subtokens_mask]

    accuracy = sum(labels == preds) / labels.shape[0]
    logger.info(f'Accuracy: {accuracy}')

    i = 0
    if preds.shape[0] > 21:
        i = random.randint(0, preds.shape[0] - 21)
    logger.info("Sampled preds: [%s]" % list2str(preds[i:i+20]))
    logger.info("Sampled labels: [%s]" % list2str(labels[i:i+20]))

    logger.info(classification_report(labels, preds, target_names=label_ids))
    return dict({'Accuracy': accuracy})
