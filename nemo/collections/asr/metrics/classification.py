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

from typing import Any, List, Optional, Tuple

import torch
from pytorch_lightning.metrics import TensorMetric
from pytorch_lightning.metrics.classification import Accuracy

__all__ = ['ClassificationAccuracy']


class ClassificationAccuracy(TensorMetric):

    """
    Computes the top-k classification accuracy provided with
    un-normalized logits of a model and ground truth targets.
    If top_k is not provided, defaults to top_1 accuracy.
    If top_k is provided as a list, then the values are sorted
    in ascending order.
    Args:
        logits: Un-normalized logits of a model. Softmax will be
            applied to these logits prior to computation of accuracy.
        targets: Vector of integers which represent indices of class
            labels.
        top_k: Optional list of integers in the range [1, max_classes].
    Returns:
        A list of length `top_k`, where each value represents top_i
        accuracy (i in `top_k`).
    """

    def __init__(self):
        super(ClassificationAccuracy, self).__init__(name="accuracy")
        self.num_samples = 0
        self.correct_counts = 0
        self.metric = Accuracy()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> List[Any]:

        batch_size = logits.shape[0]
        self.num_samples += batch_size

        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            targets = targets

            acc = self.metric(predictions, targets)
            self.correct_counts += acc * batch_size

        return (acc).to(predictions.device)
