# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List

try:
    from nemo.collections.asr.parts.k2.graph_transducer import GraphFactorizedTransducerMSELoss
    from nemo.core.utils.k2_guard import k2
except (ImportError, ModuleNotFoundError):
    pytest.skip("k2 is not installed, skipping Graph-RNNT tests.", allow_module_level=True)

EPS_SM_INPUT = 1e-6
EPS_L_INPUT = 1e-4

DEVICES = ['cpu']

if torch.cuda.is_available() and k2.with_cuda:
    DEVICES.append('cuda')


class TestGraphFactorizedTransducerMSELoss:
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_dummy(self, device):
        criterion = GraphFactorizedTransducerMSELoss()
        batch_size = 1
        enc_length = 3
        target_length = 5  # +1 blank last
        num_features = 10
        blank_logits = torch.rand((batch_size, enc_length, target_length + 1), device=device)  # -100.0
        predictions = torch.rand((batch_size, enc_length, target_length + 1, num_features), device=device)
        targets = torch.rand((batch_size, target_length, num_features), device=device)
        logits_lengths = torch.tensor([enc_length])
        targets_lengths = torch.tensor([target_length])
        loss = criterion(blank_logits, predictions, targets, logits_lengths, targets_lengths)
        assert loss.item() > 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_random_first_frame_to_all(self, device):
        criterion = GraphFactorizedTransducerMSELoss()
        batch_size = 1
        enc_length = 3
        target_length = 5  # +1 blank last
        num_features = 10
        blank_logits = torch.rand((batch_size, enc_length, target_length + 1), device=device)
        predictions = torch.rand((batch_size, enc_length, target_length + 1, num_features), device=device)
        targets = torch.rand((batch_size, target_length, num_features), device=device)
        predictions[:, 0, :-1] = targets.detach()
        blank_logits[:, 0] = -100
        blank_logits[:, :, -1] = 100
        logits_lengths = torch.tensor([enc_length])
        targets_lengths = torch.tensor([target_length])
        loss = criterion(blank_logits, predictions, targets, logits_lengths, targets_lengths)
        assert loss.item() < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_random_last_frame_to_all(self, device):
        criterion = GraphFactorizedTransducerMSELoss()
        batch_size = 1
        enc_length = 3
        target_length = 5  # +1 blank last
        num_features = 10
        blank_logits = torch.rand((batch_size, enc_length, target_length + 1), device=device)
        predictions = torch.rand((batch_size, enc_length, target_length + 1, num_features), device=device)
        targets = torch.rand((batch_size, target_length, num_features), device=device)
        predictions[:, -1, :-1] = targets.detach()
        blank_logits[:, :-1, 0] = 100.0
        blank_logits[:, -1] = -100.0
        blank_logits[:, -1, -1] = 100.0
        logits_lengths = torch.tensor([enc_length])
        targets_lengths = torch.tensor([target_length])
        loss = criterion(blank_logits, predictions, targets, logits_lengths, targets_lengths)
        assert loss.item() < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("alignment", [[0, 0, 2, 2, 2], [0, 1, 2], [0, 2]])
    def test_random_with_alignment(self, device, alignment: List[int]):
        alignment = [0, 1, 2]
        criterion = GraphFactorizedTransducerMSELoss()
        batch_size = 1
        enc_length = alignment[-1] + 1  # if not (more enc steps) - should modify algo (blank prob)
        target_length = len(alignment)  # +1 blank last
        num_features = 10
        blank_logits = torch.rand((batch_size, enc_length, target_length + 1), device=device)
        predictions = torch.rand((batch_size, enc_length, target_length + 1, num_features), device=device)
        targets = torch.rand((batch_size, target_length, num_features), device=device)
        for target_i, encoder_i in enumerate(alignment):
            predictions[0, encoder_i, target_i] = targets[0, target_i].detach()
            blank_logits[0, encoder_i, target_i] = -100
            if target_i == 0:
                continue
            prev_encoder_i = alignment[target_i - 1]
            if prev_encoder_i == encoder_i:
                blank_logits[0, target_i - 1, encoder_i] = -100
            else:
                for cur_encoder_i in range(prev_encoder_i, encoder_i):
                    blank_logits[0, cur_encoder_i, target_i] = 100
        assert len(alignment) > 0
        blank_logits[:, -1, -1] = 100.0  # last blank
        logits_lengths = torch.tensor([enc_length])
        targets_lengths = torch.tensor([target_length])
        loss = criterion(blank_logits, predictions, targets, logits_lengths, targets_lengths)
        print(loss)
