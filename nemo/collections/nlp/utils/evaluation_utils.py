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

from nemo import logging


def analyze_confusion_matrix(cm, dict, max_pairs=10):
    """
    Sort all confusions in the confusion matrix by value and display results.
    Print results in a format: (name -> name, value)
    Args:
        cm: Confusion matrix
        dict: Dictionary with key as a name and index as a value (Intents or Slots)
        max_pairs: Max number of confusions to print
    """
    threshold = 5  # just arbitrary value to take confusion with at least this number
    confused_pairs = {}
    size = cm.shape[0]
    for i in range(size):
        res = cm[i].argsort()
        for j in range(size):
            pos = res[size - j - 1]
            # no confusion - same row and column
            if pos == i:
                continue
            elif cm[i][pos] >= threshold:
                str = f'{dict[i]} -> {dict[pos]}'
                confused_pairs[str] = cm[i][pos]
            else:
                break

    # sort by max confusions and print first max_pairs
    sorted_confused_pairs = sorted(confused_pairs.items(), key=lambda x: x[1], reverse=True)
    for i, pair_str in enumerate(sorted_confused_pairs):
        if i >= max_pairs:
            break
        logging.info(pair_str)


def errors_per_class(cm, dict):
    """
    Summarize confusions per each class in the confusion matrix.
    It can be useful both for Intents and Slots.
    It counts each confusion twice in both directions.
    Args:
        cm: Confusion matrix
        dict: Dictionary with key as a name and index as a value (Intents or Slots)
    """
    size = cm.shape[0]
    confused_per_class = {}
    total_errors = 0
    for class_num in range(size):
        sum = 0
        for i in range(size):
            if i != class_num:
                sum += cm[class_num][i]
                sum += cm[i][class_num]
        confused_per_class[dict[class_num]] = sum
        total_errors += sum
        # logging.info(f'{dict[class_num]} - {sum}')

    logging.info(f'Total errors (multiplied by 2): {total_errors}')
    sorted_confused_per_class = sorted(confused_per_class.items(), key=lambda x: x[1], reverse=True)
    for conf_str in sorted_confused_per_class:
        logging.info(conf_str)


def log_misclassified_queries(intent_labels, intent_preds, queries, intent_dict, limit=50):
    """
    Display examples of Intent mistakes.
    In a format: Query, predicted and labeled intent names.
    """
    logging.info(f'*** Misclassified intent queries (limit {limit}) ***')
    cnt = 0
    for i in range(len(intent_preds)):
        if intent_labels[i] != intent_preds[i]:
            query = queries[i].split('\t')[0]
            logging.info(
                f'{query} (predicted: {intent_dict[intent_preds[i]]} - labeled: {intent_dict[intent_labels[i]]})'
            )
            cnt = cnt + 1
            if cnt >= limit:
                break


def log_misclassified_slots(
    intent_labels, intent_preds, slot_labels, slot_preds, subtokens_mask, queries, intent_dict, slot_dict, limit=50
):
    """
    Display examples of Slot mistakes.
    In a format: Query, predicted and labeled intent names and list of predicted and labeled slot numbers.
    also prints dictionary of the slots at the start for easier reading.
    """
    logging.info('')
    logging.info(f'*** Misclassified slots queries (limit {limit}) ***')
    # print slot dictionary
    logging.info(f'Slot dictionary:')
    str = ''
    for i, slot in enumerate(slot_dict):
        str += f'{i} - {slot}, '
        if i % 5 == 4 or i == len(slot_dict) - 1:
            logging.info(str)
            str = ''

    logging.info('----------------')
    cnt = 0
    for i in range(len(intent_preds)):
        cur_slot_pred = slot_preds[i][subtokens_mask[i]]
        cur_slot_label = slot_labels[i][subtokens_mask[i]]
        if not np.all(cur_slot_pred == cur_slot_label):
            query = queries[i].split('\t')[0]
            logging.info(
                f'{query} (predicted: {intent_dict[intent_preds[i]]} - labeled: {intent_dict[intent_labels[i]]})'
            )
            logging.info(f'p: {cur_slot_pred}')
            logging.info(f'l: {cur_slot_label}')
            cnt = cnt + 1
            if cnt >= limit:
                break


def check_problematic_slots(slot_preds_list, slot_dict):
    """ Check non compliance of B- and I- slots for datasets that use such slot encoding. """
    cnt = 0

    # for sentence in slot_preds:
    # slots = sentence.split(" ")
    sentence = slot_preds_list
    for i in range(len(sentence)):
        slot_name = slot_dict[int(sentence[i])]
        if slot_name.startswith("I-"):
            prev_slot_name = slot_dict[int(sentence[i - 1])]
            if slot_name[2:] != prev_slot_name[2:]:
                print("Problem: " + slot_name + " - " + prev_slot_name)
                cnt += 1
    print("Total problematic slots: " + str(cnt))
