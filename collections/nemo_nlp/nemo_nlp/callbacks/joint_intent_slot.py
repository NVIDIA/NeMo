# Copyright (c) 2019 NVIDIA Corporation
import os
import random
import time

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from nemo.utils.exp_logging import get_logger
matplotlib.use("TkAgg")


logger = get_logger('')


def tensor2list(tensor):
    return tensor.detach().cpu().tolist()


def eval_iter_callback(tensors,
                       global_vars,
                       eval_data_layer):
    if "all_intent_preds" not in global_vars.keys():
        global_vars["all_intent_preds"] = []
    if "all_intent_labels" not in global_vars.keys():
        global_vars["all_intent_labels"] = []
    if "all_slot_preds" not in global_vars.keys():
        global_vars["all_slot_preds"] = []
    if "all_slot_labels" not in global_vars.keys():
        global_vars["all_slot_labels"] = []

    intent_logits_lists, intent_labels_lists = [], []
    slot_logits_lists, slot_labels_lists = [], []

    for kv, v in tensors.items():
        if kv.startswith('intent_logits'):
            for v_tensor in v:
                for logit_tensor in v_tensor:
                    intent_logits_lists.append(tensor2list(logit_tensor))

        if kv.startswith('intents'):
            for v_tensor in v:
                for label_tensor in v_tensor:
                    intent_labels_lists.append(tensor2list(label_tensor))

        if kv.startswith('slot_logits'):
            for v_tensor in v:
                for logit_tensor in v_tensor:
                    slot_logits_lists.append(tensor2list(logit_tensor))

        if kv.startswith('slots'):
            for v_tensor in v:
                for label_tensor in v_tensor:
                    slot_labels_lists.extend(tensor2list(label_tensor))

    intent_preds = list(np.argmax(np.asarray(intent_logits_lists), 1))
    slot_preds = list(np.argmax(np.asarray(slot_logits_lists), 2).flatten())
    global_vars["all_intent_preds"].extend(intent_preds)
    global_vars["all_intent_labels"].extend(intent_labels_lists)
    global_vars["all_slot_preds"].extend(slot_preds)
    global_vars["all_slot_labels"].extend(slot_labels_lists)


def list2str(l):
    return ' '.join([str(j) for j in l])


def eval_epochs_done_callback(global_vars, graph_fold):
    intent_labels = np.asarray(global_vars['all_intent_labels'])
    intent_preds = np.asarray(global_vars['all_intent_preds'])
    correct_preds = sum(intent_labels == intent_preds)
    intent_accuracy = correct_preds / intent_labels.shape[0]
    logger.info(f'Intent accuracy: {intent_accuracy}')

    slot_labels = np.asarray(global_vars['all_slot_labels'])
    slot_preds = np.asarray(global_vars['all_slot_preds'])
    slot_accuracy = sum(slot_labels == slot_preds) / slot_labels.shape[0]
    logger.info(f'Slot accuracy: {slot_accuracy}')

    i = 0
    if intent_preds.shape[0] > 21:
        i = random.randint(0, intent_preds.shape[0] - 21)
    logger.info("Sampled i_preds: [%s]" % list2str(intent_preds[i:i+20]))
    logger.info("Sampled intents: [%s]" % list2str(intent_labels[i:i+20]))
    logger.info("Sampled s_preds: [%s]" % list2str(slot_preds[i:i+20]))
    logger.info("Sampled slots: [%s]" % list2str(slot_labels[i:i+20]))
    cm = confusion_matrix(intent_labels, intent_preds)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs(graph_fold, exist_ok=True)
    plt.savefig(os.path.join(graph_fold, time.strftime('%Y%m%d-%H%M%S')))

    logger.info(classification_report(intent_labels, intent_preds))

    return dict({'intent_accuracy': intent_accuracy,
                 'slot_accuracy': slot_accuracy})
