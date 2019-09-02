# Copyright (c) 2019 NVIDIA Corporation
import os
import random
import time

import logging

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

matplotlib.use("TkAgg")
logger = logging.getLogger('log')


def eval_iter_callback(tensors,
                       global_vars,
                       eval_data_layer):
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

    preds = list(np.argmax(np.asarray(logits_lists), 1))
    global_vars["all_preds"].extend(preds)
    global_vars["all_labels"].extend(labels_lists)


def list2str(l):
    return ' '.join([str(j) for j in l])


def eval_epochs_done_callback(global_vars, graph_fold):
    labels = np.asarray(global_vars['all_labels'])
    preds = np.asarray(global_vars['all_preds'])
    accuracy = sum(labels == preds) / labels.shape[0]
    logger.info(f'Accuracy: {accuracy}')
    i = 0
    if preds.shape[0] > 21:
        i = random.randint(0, preds.shape[0] - 21)
    logger.info("Sampled preds: [%s]" % list2str(preds[i:i+20]))
    logger.info("Sampled labels: [%s]" % list2str(labels[i:i+20]))
    cm = confusion_matrix(labels, preds)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs(graph_fold, exist_ok=True)
    plt.savefig(os.path.join(graph_fold, time.strftime('%Y%m%d-%H%M%S')))

    logger.info(classification_report(labels, preds))

    return dict({"accuracy": accuracy})
