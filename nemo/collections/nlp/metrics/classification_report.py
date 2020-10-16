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

from typing import Any, Dict, Optional

import torch
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.utils import METRIC_EPS

from nemo.utils import logging

__all__ = ['ClassificationReport']


class ClassificationReport(Metric):
    """
    This metric computes the number of True Positive, False Negative, and False Positive examples per class.
    When doing distributed training/evaluation the result of res=ClassificationReport(predictions, labels) calls
    will be all-reduced between all workers using SUM operations.

    If used with PytorchLightning LightningModule, include TPs, FNs, and FPs inside validation_step results.
    Then aggregate them at the end of validation epoch to correctly compute validation precision, recall, f1
    using get_precision_recall_f1().

    Example:
        def validation_step(self, batch, batch_idx):
            ...
            tp, fp, fn = self.punct_class_report(predictions, labels)
            return {'val_loss': loss_value, 'val_tp': tp, 'val_fp': fp, 'val_fn': fn}

        def validation_epoch_end(self, outputs):
            ...
            tp = torch.sum(torch.stack([x['tp'] for x in outputs]), 0)
            fn = torch.sum(torch.stack([x['fn'] for x in outputs]), 0)
            fp = torch.sum(torch.stack([x['fp'] for x in outputs]), 0)
            precision, recall, f1 = self.get_precision_recall_f1(tp, fn, fp, mode='macro')
            tensorboard_logs = {'validation_loss': avg_loss, 'precision': precision, 'f1': f1, 'recall': recall}
            return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    Args:
        num_classes: number of classes in the dataset
        label_ids (optional): label name to label id mapping
    Returns:
        res: a torch.Tensor object with three elements: [true positives, false positives, false negatives].
    """

    def __init__(
        self,
        num_classes: int,
        label_ids: Dict[str, int] = None,
        mode: str = 'macro',
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, process_group=process_group)
        self.num_classes = num_classes
        if label_ids:
            self.ids_to_labels = {v: k for k, v in label_ids.items()}
        else:
            self.ids_to_labels = None
        self.mode = mode

        self.add_state("tp", default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state("fn", default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state("fp", default=torch.zeros(num_classes), dist_reduce_fx='sum')

    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        TP = []
        FN = []
        FP = []
        for label_id in range(self.num_classes):
            current_label = labels == label_id
            label_predicted = predictions == label_id

            TP.append((label_predicted == current_label)[label_predicted].sum())
            FP.append((label_predicted != current_label)[label_predicted].sum())
            FN.append((label_predicted != current_label)[current_label].sum())

        self.tp = torch.tensor(TP).to(predictions.device)
        self.fn = torch.tensor(FN).to(predictions.device)
        self.fp = torch.tensor(FP).to(predictions.device)

    def compute(self):
        """
        Calculates and logs classification report similar to sklearn.metrics.classification_report
        Args:
            tp: Number of true positives per class
            fn: Number of false negatives per class
            fp: Number of false positives per class
            mode: 'macro' to use macro averaging to combine f1 scores for classes
        Return:
            aggregated precision, recall, f1
        """
        num_examples_per_class = self.tp + self.fn
        total_examples = torch.sum(num_examples_per_class)
        num_non_empty_classes = torch.nonzero(num_examples_per_class).size(0)

        precision = torch.true_divide(self.tp * 100, (self.tp + self.fp + METRIC_EPS))
        recall = torch.true_divide(self.tp * 100, (self.tp + self.fn + METRIC_EPS))
        f1 = torch.true_divide(2 * precision * recall, (precision + recall + METRIC_EPS))

        report = '\n{:50s}   {:10s}   {:10s}   {:10s}   {:10s}'.format('label', 'precision', 'recall', 'f1', 'support')
        for i in range(self.tp.shape[0]):
            label = f'label_id: {i}'
            if self.ids_to_labels and i in self.ids_to_labels:
                label = f'{self.ids_to_labels[i]} ({label})'

            report += '\n{:50s}   {:8.2f}   {:8.2f}   {:8.2f}   {:8.0f}'.format(
                label, precision[i], recall[i], f1[i], num_examples_per_class[i]
            )

        micro_precision = torch.true_divide(torch.sum(self.tp) * 100, torch.sum(self.tp + self.fp) + METRIC_EPS)
        micro_recall = torch.true_divide(torch.sum(self.tp) * 100, torch.sum(self.tp + self.fn) + METRIC_EPS)
        micro_f1 = torch.true_divide(2 * micro_precision * micro_recall, (micro_precision + micro_recall + METRIC_EPS))

        macro_precision = torch.sum(precision) / num_non_empty_classes
        macro_recall = torch.sum(recall) / num_non_empty_classes
        macro_f1 = torch.sum(f1) / num_non_empty_classes

        weighted_precision = torch.sum(precision * num_examples_per_class) / total_examples
        weighted_recall = torch.sum(recall * num_examples_per_class) / total_examples
        weighted_f1 = torch.sum(f1 * num_examples_per_class) / total_examples

        report += "\n-------------------"

        report += '\n{:50s}   {:8.2f}   {:8.2f}   {:8.2f}   {:8.0f}'.format(
            'micro avg', micro_precision, micro_recall, micro_f1, total_examples
        )

        report += '\n{:50s}   {:8.2f}   {:8.2f}   {:8.2f}   {:8.0f}'.format(
            'macro avg', macro_precision, macro_recall, macro_f1, total_examples
        )
        report += (
            '\n{:50s}   {:8.2f}   {:8.2f}   {:8.2f}   {:8.0f}'.format(
                'weighted avg', weighted_precision, weighted_recall, weighted_f1, total_examples
            )
            + '\n'
        )

        logging.info(report)

        if self.mode == 'macro':
            return macro_precision, macro_recall, macro_f1
        elif self.mode == 'weighted':
            return weighted_precision, weighted_recall, weighted_f1
        elif self.mode == 'micro':
            return micro_precision, micro_recall, micro_f1
        elif self.mode == 'all':
            return precision, recall, f1
        else:
            raise ValueError(
                f'{self.mode} mode is not supported. Choose "macro" to get aggregated numbers \
            or "all" to get values for each class.'
            )
