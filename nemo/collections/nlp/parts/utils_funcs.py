# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

__all__ = ['list2str', 'tensor2list', 'plot_confusion_matrix', 'get_classification_report']

import os
import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch import Tensor

from nemo.utils import logging


def list2str(l: List[int]) -> str:
    """ Converts list to a string"""
    return ' '.join([str(x) for x in l])


def tensor2list(tensor: Tensor) -> List[Union[int, float]]:
    """ Converts tensor to a list """
    return tensor.detach().cpu().tolist()


def plot_confusion_matrix(
    labels: List[int],
    preds: List[int],
    graph_fold: str,
    label_ids: Dict[str, int] = None,
    normalize: bool = False,
    prefix: str = '',
):
    '''
    Plot confusion matrix.
    Args:
      labels: ground truth labels
      preds: model predictions
      graph_fold: path to a folder to store confusion matrix plot
      label_ids: str label to id map, for example: {'O': 0, 'LOC': 1}
      normalize: whether to normalize confusion matrix
      prefix: prefix for the plot name
    '''
    if label_ids is None:
        _plot_confusion_matrix(labels, preds, graph_fold)

    else:
        # remove labels from label_ids that don't appear in the dev set
        used_labels = set(labels) | set(preds)
        label_ids = {k: label_ids[k] for k, v in label_ids.items() if v in used_labels}

        ids_to_labels = {label_ids[k]: k for k in label_ids}
        classes = [ids_to_labels[id] for id in sorted(label_ids.values())]

        title = 'Confusion_matrix'
        cm = confusion_matrix(labels, preds)
        if normalize:
            sums = cm.sum(axis=1)[:, np.newaxis]
            sums = np.where(sums == 0, 1, sums)
            cm = cm.astype('float') / sums
            title = 'Normalized_' + title

        fig = plt.figure()
        ax = fig.add_subplot(111)

        cax = ax.matshow(cm)

        ax.set_xticks(np.arange(-1, len(classes)))
        ax.set_yticks(np.arange(-1, len(classes)))
        ax.set_xticklabels([''] + classes, rotation=90)
        ax.set_yticklabels([''] + classes)
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')

        os.makedirs(graph_fold, exist_ok=True)
        fig.colorbar(cax)

        title = (prefix + title).strip()
        fig_name = os.path.join(graph_fold, title + '_' + time.strftime('%Y%m%d-%H%M%S'))
        plt.savefig(fig_name)
        logging.info(f'Confusion matrix saved to {fig_name}')


def _plot_confusion_matrix(labels: List[int], preds: List[int], graph_fold: str):
    """
    Plot confusion matrix
    Args:
        labels: ground truth labels
        preds: model predictions
        graph_fold: path to a folder to store confusion matrix plot
    """
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


def get_classification_report(labels, preds, label_ids, output_dict=False):
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

    return classification_report(labels, preds, target_names=labels_names, digits=4, output_dict=output_dict)


def is_last_rank():
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def get_last_rank():
    return torch.distributed.get_world_size() - 1


# Temporarily putting microbatch calculators here instead of importing them from Apex.
# This is due to a bug that was fixed here: https://github.com/NVIDIA/apex/commit/7d9038782b7a39c7dd05201e17586fd38b7a7c94
# We can remove these once the above commit is available in the NGC PyTorch container.
try:
    import apex.transformer.pipeline_parallel.utils as apex_utils
    from apex.transformer.pipeline_parallel.utils import _ensure_var_is_not_initialized
    from apex.transformer.microbatches import ConstantNumMicroBatches, RampupBatchsizeNumMicroBatches

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


def setup_microbatch_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
) -> None:
    if not HAVE_APEX:
        raise ImportError(
            "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
        )
    _ensure_var_is_not_initialized(apex_utils._GLOBAL_NUM_MICROBATCHES_CALCULATOR, 'num microbatches calculator')

    apex_utils._GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(
        rank, rampup_batch_size, global_batch_size, micro_batch_size, data_parallel_size
    )


def _reconfigure_microbatch_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
) -> None:
    if not HAVE_APEX:
        raise ImportError(
            "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
        )
    apex_utils._GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(
        rank, rampup_batch_size, global_batch_size, micro_batch_size, data_parallel_size
    )


def build_num_microbatches_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
):
    if not HAVE_APEX:
        raise ImportError(
            "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
        )
    # Constant num micro-batches.
    if rampup_batch_size is None:
        num_microbatches_calculator = ConstantNumMicroBatches(global_batch_size, micro_batch_size, data_parallel_size)
        if rank == 0:
            logging.info("setting number of micro-batches to constant {}".format(num_microbatches_calculator.get()))

    else:
        assert len(rampup_batch_size) == 3, (
            "expected the following "
            "format: --rampup-batch-size <start batch size> "
            "<batch size incerement> <ramp-up samples>"
        )
        start_batch_size = int(rampup_batch_size[0])
        batch_size_increment = int(rampup_batch_size[1])
        ramup_samples = int(rampup_batch_size[2])
        if rank == 0:
            logging.info(
                "will use batch size rampup starting from global batch "
                "size {} to global batch size {} with batch size increments "
                "{} over {} samples.".format(start_batch_size, global_batch_size, batch_size_increment, ramup_samples),
            )
        num_microbatches_calculator = RampupBatchsizeNumMicroBatches(
            start_batch_size,
            batch_size_increment,
            ramup_samples,
            global_batch_size,
            micro_batch_size,
            data_parallel_size,
        )

    return num_microbatches_calculator
