# Copyright (c) 2019 NVIDIA Corporation

import os
import random
import time

import matplotlib
from matplotlib import pyplot as plt
import nemo
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']


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
    if "all_subtokens_mask" not in global_vars.keys():
        global_vars["all_subtokens_mask"] = []

    all_intent_logits, all_intent_labels = [], []
    all_slot_logits, all_slot_labels = [], []
    all_subtokens_mask = []
    for kv, v in tensors.items():
        if kv.startswith('intent_logits'):
            for v_tensor in v:
                for logit_tensor in v_tensor:
                    all_intent_logits.append(tensor2list(logit_tensor))

        if kv.startswith('intents'):
            for v_tensor in v:
                for label_tensor in v_tensor:
                    all_intent_labels.append(tensor2list(label_tensor))

        if kv.startswith('slot_logits'):
            for v_tensor in v:
                for logit_tensor in v_tensor:
                    all_slot_logits.append(tensor2list(logit_tensor))

        if kv.startswith('slots'):
            for v_tensor in v:
                for label_tensor in v_tensor:
                    all_slot_labels.extend(tensor2list(label_tensor))

        if kv.startswith('subtokens_mask'):
            for v_tensor in v:
                for subtokens_mask_tensor in v_tensor:
                    all_subtokens_mask.extend(
                        tensor2list(subtokens_mask_tensor))

    all_intent_preds = list(np.argmax(np.asarray(all_intent_logits), 1))
    all_slot_preds = list(np.argmax(np.asarray(all_slot_logits), 2).flatten())
    global_vars["all_intent_preds"].extend(all_intent_preds)
    global_vars["all_intent_labels"].extend(all_intent_labels)
    global_vars["all_slot_preds"].extend(all_slot_preds)
    global_vars["all_slot_labels"].extend(all_slot_labels)
    global_vars["all_subtokens_mask"].extend(all_subtokens_mask)


def list2str(l):
    return ' '.join([str(j) for j in l])


def eval_epochs_done_callback(global_vars, graph_fold):
    intent_labels = np.asarray(global_vars['all_intent_labels'])
    intent_preds = np.asarray(global_vars['all_intent_preds'])

    slot_labels = np.asarray(global_vars['all_slot_labels'])
    slot_preds = np.asarray(global_vars['all_slot_preds'])
    subtokens_mask = np.asarray(global_vars['all_subtokens_mask']) > 0.5

    slot_labels = slot_labels[subtokens_mask]
    slot_preds = slot_preds[subtokens_mask]

    i = 0
    if intent_preds.shape[0] > 21:
        i = random.randint(0, intent_preds.shape[0] - 21)
    nemo.logging.info("Sampled i_preds: [%s]" % list2str(intent_preds[i:i+20]))
    nemo.logging.info(
        "Sampled intents: [%s]" % list2str(intent_labels[i:i+20]))
    nemo.logging.info("Sampled s_preds: [%s]" % list2str(slot_preds[i:i+20]))
    nemo.logging.info("Sampled slots: [%s]" % list2str(slot_labels[i:i+20]))
    cm = confusion_matrix(intent_labels, intent_preds)
    nemo.logging.info(f'Confusion matrix:\n{cm}')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs(graph_fold, exist_ok=True)
    plt.savefig(os.path.join(graph_fold, time.strftime('%Y%m%d-%H%M%S')))

    nemo.logging.info('Intent prediction results')
    correct_preds = sum(intent_labels == intent_preds)
    intent_accuracy = correct_preds / intent_labels.shape[0]
    nemo.logging.info(f'Intent accuracy: {intent_accuracy}')
    nemo.logging.info(f'Classification report:\n \
        {classification_report(intent_labels, intent_preds)}')

    nemo.logging.info('Slot prediction results')
    slot_accuracy = sum(slot_labels == slot_preds) / slot_labels.shape[0]
    nemo.logging.info(f'Slot accuracy: {slot_accuracy}')
    nemo.logging.info(f'Classification report:\n \
        {classification_report(slot_labels[:-2], slot_preds[:-2])}')

    return dict({'intent_accuracy': intent_accuracy,
                 'slot_accuracy': slot_accuracy})
