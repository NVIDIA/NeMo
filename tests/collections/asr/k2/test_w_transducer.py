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

try:
    from nemo.collections.asr.parts.k2.w_transducer import GraphWTransducerLoss
    from nemo.core.utils.k2_guard import k2
except (ImportError, ModuleNotFoundError):
    pytest.skip("k2 is not installed, skipping Graph-W-Transducer tests.", allow_module_level=True)

DEVICES = ['cpu']

if torch.cuda.is_available() and k2.with_cuda:
    DEVICES.append('cuda')


class TestGraphWTransducerLoss:
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("blank_first", [True, False])
    @pytest.mark.parametrize("last_blank_mode", ["allow_ignore", "force_last"])
    @pytest.mark.parametrize("graph_mode", ["sequential", "skip_frames"])
    def test_grid_compose_equivalent(
        self, device: torch.device, blank_first: bool, last_blank_mode, graph_mode, rnn_loss_sample_data
    ):
        # TODO:  "force_all" mode fully implemented yet
        sample_data = rnn_loss_sample_data.get_sample_small_random(blank_first, device=device)
        criterion = GraphWTransducerLoss(
            blank=sample_data.blank_id,
            last_blank_mode=last_blank_mode,
            graph_mode=graph_mode,
            connect_composed=True,
            use_grid_implementation=False,
        )
        text_tensor = sample_data.targets[0]
        sequence_length = sample_data.logits.shape[1]
        graph_grid = criterion.get_grid(text_tensor, sequence_length, sample_data.num_labels)
        graph_composed = criterion.get_composed_lattice(text_tensor, sequence_length, sample_data.num_labels)
        assert k2.is_rand_equivalent(
            graph_grid, graph_composed, log_semiring=True, treat_epsilons_specially=False
        ), "Grid and composed graphs are not equivalent."
