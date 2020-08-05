# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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

from typing import Dict, List

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef

__all__ = ['compute_metrics']


def accuracy(preds: List[int], labels: List[int]):
    return {"acc": (preds == labels).mean()}


def acc_and_f1(preds: List[int], labels: List[int]):
    accuracy = (preds == labels).mean()
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {"acc": accuracy, "f1": f1}


def mcc(preds: List[int], labels: List[int]):
    return {"mcc": matthews_corrcoef(labels, preds)}


def pearson_and_spearman(preds: List[int], labels: List[int]):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {"pearson": pearson_corr, "spearmanr": spearman_corr, "pear+spear av": (pearson_corr + spearman_corr) / 2}


def compute_metrics(task_name: str, preds: List[int], labels: List[int]) -> Dict[str, float]:
    """
    Computes metrics for GLUE tasks
    Args:
        task_name: GLUE task name
        preds: model predictions
        labels: golden labels
    Returns:
        metrics
    """
    if len(preds) != len(labels):
        raise ValueError("Predictions and labels must have the same length")

    metric_fn = accuracy
    if task_name == 'cola':
        metric_fn = mcc
    elif task_name in ['mrpc', 'qqp']:
        metric_fn = acc_and_f1
    elif task_name == 'sts-b':
        metric_fn = pearson_and_spearman

    return metric_fn(preds, labels)
