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

from typing import List

import pytest
import torch

from nemo.collections.asr.parts.utils.rnnt_utils import BatchedHyps

DEVICES: List[torch.device] = [torch.device("cpu")]

if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICES.append(torch.device("mps"))


class TestBatchedHyps:
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_intantiate(self, device: torch.device):
        hyps = BatchedHyps(batch_size=2, init_length=3, device=device)
        assert torch.is_tensor(hyps.timesteps)
        # device: for mps device we need to use `type`, not directly compare
        assert hyps.timesteps.device.type == device.type
        assert hyps.timesteps.shape == (2, 3)

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [-1, 0])
    def test_intantiate_incorrect_batch_size(self, batch_size):
        with pytest.raises(ValueError):
            _ = BatchedHyps(batch_size=batch_size, init_length=3)

    @pytest.mark.unit
    @pytest.mark.parametrize("init_length", [-1, 0])
    def test_intantiate_incorrect_init_length(self, init_length):
        with pytest.raises(ValueError):
            _ = BatchedHyps(batch_size=1, init_length=init_length)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_add_results(self, device: torch.device):
        pass  # TODO: implement

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_add_multiple_results(self, device: torch.device):
        pass  # TODO: implement

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_convert_to_hyps(self, device: torch.device):
        pass  # TODO: implement

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_torch_jit_compatibility(self, device: torch.device):
        @torch.jit.script
        def hyps_add_wrapper(
            active_indices: torch.Tensor, labels: torch.Tensor, time_indices: torch.Tensor, scores: torch.Tensor
        ):
            hyps = BatchedHyps(batch_size=2, init_length=3, device=active_indices.device)
            hyps.add_results_(active_indices=active_indices, labels=labels, time_indices=time_indices, scores=scores)
            return hyps

        scores = torch.tensor([0.1, 0.1], device=device)
        hyps = hyps_add_wrapper(
            torch.tensor([0, 1], device=device),
            torch.tensor([2, 4], device=device),
            torch.tensor([0, 0], device=device),
            scores,
        )
        assert torch.allclose(hyps.scores, scores)
