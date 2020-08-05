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

from typing import Dict

import torch
from pytorch_lightning.metrics import TensorMetric

from nemo.utils import logging

__all__ = ['ClassificationReport']


class ClassificationReport(TensorMetric):
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

    def __init__(self, num_classes: int, label_ids: Dict[str, int] = None):
        super(ClassificationReport, self).__init__(name="ClassificationReport")
        self.num_classes = num_classes
        if label_ids:
            self.ids_to_labels = {v: k for k, v in label_ids.items()}
        else:
            self.ids_to_labels = None

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        TP = []
        FN = []
        FP = []
        for label_id in range(self.num_classes):
            current_label = labels == label_id
            label_predicted = predictions == label_id

            TP.append((label_predicted == current_label)[label_predicted].sum())
            FP.append((label_predicted != current_label)[label_predicted].sum())
            FN.append((label_predicted != current_label)[current_label].sum())
        return torch.tensor([TP, FP, FN]).to(predictions.device)

    def get_precision_recall_f1(
        self, tp: torch.Tensor, fn: torch.Tensor, fp: torch.Tensor, mode='macro',
    ):
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
        zeros = torch.zeros_like(tp)
        num_classes = tp.shape[0]
        num_examples_per_class = tp + fn
        total_examples = torch.sum(num_examples_per_class)

        precision = torch.where(tp + fp != zeros, tp / (tp + fp) * 100, zeros)
        recall = torch.where(tp + fn != zeros, tp / (tp + fn) * 100, zeros)
        f1 = torch.where(precision + recall != zeros, 2 * precision * recall / (precision + recall), zeros)

        report = '\n{:50s}   {:10s}   {:10s}   {:10s}   {:10s}'.format('label', 'precision', 'recall', 'f1', 'support')
        for id in range(tp.shape[0]):
            label = f'label_id: {id}'
            if self.ids_to_labels and id in self.ids_to_labels:
                label = f'{self.ids_to_labels[id]} ({label})'

            report += '\n{:50s}   {:8.2f}   {:8.2f}   {:8.2f}   {:8.0f}'.format(
                label, precision[id], recall[id], f1[id], num_examples_per_class[id]
            )

        micro_precision = torch.where(torch.sum(tp + fp) != zeros, torch.sum(tp) / torch.sum(tp + fp) * 100, zeros)
        micro_recall = torch.where(torch.sum(tp + fn) != zeros, torch.sum(tp) / torch.sum(tp + fn) * 100, zeros)
        micro_f1 = torch.where(
            micro_precision + micro_recall != zeros,
            2 * micro_precision * micro_recall / (micro_precision + micro_recall),
            zeros,
        )

        macro_precision = torch.sum(precision) / num_classes
        macro_recall = torch.sum(recall) / num_classes
        macro_f1 = torch.sum(f1) / num_classes

        weighted_precision = torch.sum(precision * num_examples_per_class) / total_examples
        weighted_recall = torch.sum(recall * num_examples_per_class) / total_examples
        weighted_f1 = torch.sum(f1 * num_examples_per_class) / total_examples

        report += '\n{:50s}   {:8.2f}   {:8.2f}   {:8.2f}   {:8.0f}'.format(
            'micro avg', micro_precision[0], micro_recall[0], micro_f1[0], total_examples
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

        if mode == 'macro':
            return macro_precision, macro_recall, macro_f1
        elif mode == 'weighted':
            return weighted_precision, weighted_recall, weighted_f1
        elif mode == 'micro':
            return micro_precision[0], micro_recall[0], micro_f1[0]
        elif mode == 'all':
            return precision, recall, f1
        else:
            raise ValueError(
                f'{mode} mode is not supported. Choose "macro" to get aggregated numbers \
            or "all" to get values for each class.'
            )
