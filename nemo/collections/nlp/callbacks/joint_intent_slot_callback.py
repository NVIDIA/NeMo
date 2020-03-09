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
                    all_subtokens_mask.extend(tensor2list(subtokens_mask_tensor))

    all_intent_preds = list(np.argmax(np.asarray(all_intent_logits), 1))
    all_slot_preds = list(np.argmax(np.asarray(all_slot_logits), 2).flatten())
    global_vars["all_intent_preds"].extend(all_intent_preds)
    global_vars["all_intent_labels"].extend(all_intent_labels)
    global_vars["all_slot_preds"].extend(all_slot_preds)
    global_vars["all_slot_labels"].extend(all_slot_labels)
    global_vars["all_subtokens_mask"].extend(all_subtokens_mask)


def eval_epochs_done_callback(global_vars, graph_fold):
    intent_labels = np.asarray(global_vars['all_intent_labels'])
    intent_preds = np.asarray(global_vars['all_intent_preds'])

    slot_labels = np.asarray(global_vars['all_slot_labels'])
    slot_preds = np.asarray(global_vars['all_slot_preds'])
    subtokens_mask = np.asarray(global_vars['all_subtokens_mask']) > 0.5

    slot_labels = slot_labels[subtokens_mask]
    slot_preds = slot_preds[subtokens_mask]

    # print predictions and labels for a small random subset of data
    sample_size = 20
    i = 0
    if intent_preds.shape[0] > sample_size + 1:
        i = random.randint(0, intent_preds.shape[0] - sample_size - 1)
    logging.info("Sampled i_preds: [%s]" % list2str(intent_preds[i : i + sample_size]))
    logging.info("Sampled intents: [%s]" % list2str(intent_labels[i : i + sample_size]))
    logging.info("Sampled s_preds: [%s]" % list2str(slot_preds[i : i + sample_size]))
    logging.info("Sampled slots: [%s]" % list2str(slot_labels[i : i + sample_size]))

    plot_confusion_matrix(intent_labels, intent_preds, graph_fold)

    logging.info('Intent prediction results')
    correct_preds = sum(intent_labels == intent_preds)
    intent_accuracy = correct_preds / intent_labels.shape[0]
    logging.info(f'Intent accuracy: {intent_accuracy}')
    logging.info(
        f'Classification report:\n \
        {classification_report(intent_labels, intent_preds)}'
    )

    logging.info('Slot prediction results')
    slot_accuracy = sum(slot_labels == slot_preds) / slot_labels.shape[0]
    logging.info(f'Slot accuracy: {slot_accuracy}')
    logging.info(
        f'Classification report:\n \
        {classification_report(slot_labels[:-2], slot_preds[:-2])}'
    )

    return dict({'intent_accuracy': intent_accuracy, 'slot_accuracy': slot_accuracy})
