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

import torch
from pytorch_lightning.metrics import Metric

__all__ = ['TopKClassificationAccuracy']


class TopKClassificationAccuracy(Metric):
    """
    This metric computes numerator and denominator for Overall Accuracy between logits and labels.
    When doing distributed training/evaluation the result of res=TopKClassificationAccuracy(logits, labels) calls
    will be all-reduced between all workers using SUM operations.
    Here contains two numbers res=[correctly_predicted, total_samples]. Accuracy=correctly_predicted/total_samples.

    If used with PytorchLightning LightningModule, include correct_count and total_count inside validation_step results.
    Then aggregate (sum) then at the end of validation epoch to correctly compute validation WER.

    Example:
        def validation_step(self, batch, batch_idx):
            ...
            correct_count, total_count = self._accuracy(logits, labels)
            return {'val_loss': loss_value, 'val_correct_count': correct_count, 'val_total_count': total_count}

        def validation_epoch_end(self, outputs):
            ...
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            correct_counts = torch.stack([x['val_correct_counts'] for x in outputs])
            total_counts = torch.stack([x['val_total_counts'] for x in outputs])

            topk_scores = compute_topk_accuracy(correct_counts, total_counts)

            tensorboard_log = {'val_loss': val_loss_mean}
            for top_k, score in zip(self._accuracy.top_k, topk_scores):
                tensorboard_log['val_epoch_top@{}'.format(top_k)] = score

            return {'log': tensorboard_log}

    Args:
        top_k: Optional list of integers. Defaults to [1].

    Returns:
        res: a torch.Tensor object with two elements: [correct_count, total_count]. To correctly compute average
        accuracy, compute acc=correct_count/total_count
    """

    def __init__(self, top_k=None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if top_k is None:
            top_k = [1]

        self.top_k = top_k
        self.add_state("correct_counts_k", default=torch.zeros(len(self.top_k)), dist_reduce_fx='sum')
        self.add_state("total_counts_k", default=torch.zeros(len(self.top_k)), dist_reduce_fx='sum')

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            max_k = max(self.top_k)

            _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
            predictions = predictions.t()
            correct = predictions.eq(labels.view(1, -1)).expand_as(predictions)

            correct_counts_k = []
            total_counts_k = []

            for k in self.top_k:
                correct_k = correct[:k].view(-1).float().sum()
                total_k = labels.shape[0]

                correct_counts_k.append(correct_k)
                total_counts_k.append(total_k)

            self.correct_counts_k = torch.tensor(correct_counts_k, dtype=labels.dtype, device=labels.device)
            self.total_counts_k = torch.tensor(total_counts_k, dtype=labels.dtype, device=labels.device)

    def compute(self):
        """
        Computes the top-k accuracy.

        Returns:
            A list of length `K`, such that k-th index corresponds to top-k accuracy
            over all distributed processes.
        """
        if self.top_k == [1]:
            return [self.correct_counts_k.float() / self.total_counts_k]

        else:
            top_k_scores = []

            for ki in range(self.correct_counts_k.shape[-1]):
                correct_count = self.correct_counts_k[ki].item()
                total_count = self.total_counts_k[ki].item()
                top_k_scores.append(correct_count / float(total_count))

            return top_k_scores


def compute_topk_accuracy(correct_counts_k, total_counts_k):
    """
    Computes the top-k accuracy when provided with a stacked tensor from multiple
    DDP processes.
    Args:
        correct_counts: Tensor of shape [D, K], D being the number of processes
            and K being the top-k parameter.
        total_counts: Tensor of shape [D, K], D being the number of processes
            and K being the top-k parameter.
    Returns:
        A list of length `K`, such that k-th index corresponds to top-k accuracy
        over all distributed processes.
    """
    top_k_scores = []

    for ki in range(correct_counts_k.shape[-1]):
        correct_count = correct_counts_k[ki].item()
        total_count = total_counts_k[ki].item()
        top_k_scores.append(correct_count / float(total_count))

        return top_k_scores
