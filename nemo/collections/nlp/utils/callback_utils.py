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

import os
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from nemo import logging

__all__ = ['list2str', 'tensor2list', 'plot_confusion_matrix', 'tensor2numpy']


def list2str(l):
    return ' '.join([str(x) for x in l])


def tensor2list(tensor):
    return tensor.detach().cpu().tolist()


def tensor2numpy(tensor):
    return tensor.detach().cpu().numpy()


def plot_confusion_matrix(labels, preds, graph_fold, label_ids=None, normalize=False, prefix=''):
    '''
    Plot confusion matrix.
    Args:
      label_ids (dict): label to id map, for example: {'O': 0, 'LOC': 1}
      labels (list of ints): list of true labels
      preds (list of ints): list of predicted labels
      graph_fold (str): path to output folder
      normalize (bool): flag to indicate whether to normalize confusion matrix
      prefix (str): prefix for the plot name

    '''
    if label_ids is None:
        _plot_confusion_matrix(labels, preds, graph_fold)

    else:
        # remove labels from label_ids that don't appear in the dev set
        used_labels = set(labels) | set(preds)
        label_ids = {k: label_ids[k] for k, v in label_ids.items() if v in used_labels}

        ids_to_labels = {label_ids[k]: k for k in label_ids}
        classes = [ids_to_labels[id] for id in sorted(label_ids.values())]

        title = 'Confusion matrix'
        cm = confusion_matrix(labels, preds)
        if normalize:
            sums = cm.sum(axis=1)[:, np.newaxis]
            sums = np.where(sums == 0, 1, sums)
            cm = cm.astype('float') / sums
            title = 'Normalized ' + title

        fig = plt.figure()
        ax = fig.add_subplot(111)

        cax = ax.matshow(cm)
        ax.set_xticks(np.arange(-1, len(classes) + 1))
        ax.set_yticks(np.arange(-1, len(classes) + 1))
        ax.set_xticklabels([''] + classes, rotation=90)
        ax.set_yticklabels([''] + classes)
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')

        os.makedirs(graph_fold, exist_ok=True)
        fig.colorbar(cax)

        title = (prefix + ' ' + title).strip()
        plt.savefig(os.path.join(graph_fold, title + '_' + time.strftime('%Y%m%d-%H%M%S')))


def _plot_confusion_matrix(labels, preds, graph_fold):
    cm = confusion_matrix(labels, preds)
    logging.info(f'Confusion matrix:\n{cm}')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs(graph_fold, exist_ok=True)
    plt.savefig(os.path.join(graph_fold, time.strftime('%Y%m%d-%H%M%S')))


def get_classification_report(labels, preds, label_ids):
    """
    Returns classification report
    """
    # remove labels from label_ids that don't appear in predictions or ground truths
    used_labels = set(labels) | set(preds)
    labels_names = [
        k + ' (label id: ' + str(v) + ')'
        for k, v in sorted(label_ids.items(), key=lambda item: item[1])
        if v in used_labels
    ]

    return classification_report(labels, preds, target_names=labels_names, digits=4)


def get_f1_scores(labels, preds, average_modes=['binary', 'weighted', 'macro', 'micro']):
    """
    Returns a dictionary with f1_score based on different averaging mode
    Args:
      labels (list of ints): list of true labels
      preds (list of ints): list of predicted labels
      average_modes (list): list of possible averaging types. Binary for is supported only for binary target.
    """
    f1_scores = {}
    for average in average_modes:
        f1_scores['F1 ' + average] = round(f1_score(labels, preds, average=average) * 100, 2)

    return f1_scores
