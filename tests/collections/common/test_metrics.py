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

import pytest
import torch

from nemo.collections.common.metrics.classification_accuracy import TopKClassificationAccuracy, compute_topk_accuracy


class TestCommonMetrics:
    top_k_logits = torch.tensor([[0.1, 0.3, 0.2, 0.0], [0.9, 0.6, 0.2, 0.3], [0.2, 0.1, 0.4, 0.3]],)  # 1  # 0  # 2

    @pytest.mark.unit
    def test_top_1_accuracy(self):
        labels = torch.tensor([0, 0, 2], dtype=torch.long)

        accuracy = TopKClassificationAccuracy(top_k=None)
        correct, total = accuracy(logits=self.top_k_logits, labels=labels)

        assert correct.shape == torch.Size([1])
        assert total.shape == torch.Size([1])
        assert abs((correct / total) - 0.667) < 1e-3

    @pytest.mark.unit
    def test_top_1_2_accuracy(self):
        labels = torch.tensor([0, 1, 0], dtype=torch.long)

        accuracy = TopKClassificationAccuracy(top_k=[1, 2])
        correct, total = accuracy(logits=self.top_k_logits, labels=labels)

        assert correct.shape == torch.Size([2])
        assert total.shape == torch.Size([2])

        top1_acc = correct[0] / total[0]
        top2_acc = correct[1] / total[1]

        assert abs(top1_acc - 0.0) < 1e-3
        assert abs(top2_acc - 0.333) < 1e-3

    @pytest.mark.unit
    def test_top_1_accuracy_distributed(self):
        # Simulate test on 2 process DDP execution
        labels = torch.tensor([[0, 0, 2], [2, 0, 0]], dtype=torch.long)

        accuracy = TopKClassificationAccuracy(top_k=None)
        correct1, total1 = accuracy(logits=self.top_k_logits, labels=labels[0])
        correct2, total2 = accuracy(logits=torch.flip(self.top_k_logits, dims=[1]), labels=labels[1])  # reverse logits

        correct = torch.stack([correct1, correct2])
        total = torch.stack([total1, total2])

        assert correct.shape == torch.Size([2, 1])
        assert total.shape == torch.Size([2, 1])

        proc1_acc = correct[0] / total[0]
        proc2_acc = correct[1] / total[1]

        assert abs(proc1_acc - 0.667) < 1e-3  # 2/3
        assert abs(proc2_acc - 0.333) < 1e-3  # 1/3

        acc_topk = compute_topk_accuracy(correct, total)
        acc_top1 = acc_topk[0]

        assert abs(acc_top1 - 0.5) < 1e-3  # 3/6

    @pytest.mark.unit
    def test_top_1_accuracy_distributed_uneven_batch(self):
        # Simulate test on 2 process DDP execution
        accuracy = TopKClassificationAccuracy(top_k=None)
        correct1, total1 = accuracy(logits=self.top_k_logits, labels=torch.tensor([0, 0, 2]))
        correct2, total2 = accuracy(
            logits=torch.flip(self.top_k_logits, dims=[1])[:2, :],  # reverse logits, select first 2 samples
            labels=torch.tensor([2, 0]),
        )  # reduce number of labels

        correct = torch.stack([correct1, correct2])
        total = torch.stack([total1, total2])

        assert correct.shape == torch.Size([2, 1])
        assert total.shape == torch.Size([2, 1])

        proc1_acc = correct[0] / total[0]
        proc2_acc = correct[1] / total[1]

        assert abs(proc1_acc - 0.667) < 1e-3  # 2/3
        assert abs(proc2_acc - 0.500) < 1e-3  # 1/2

        acc_topk = compute_topk_accuracy(correct, total)
        acc_top1 = acc_topk[0]

        assert abs(acc_top1 - 0.6) < 1e-3  # 3/5
