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

import numpy as np
import pytest
import torch

from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_numpy import RNNTLoss as RNNTLoss_Numpy

try:
    from nemo.collections.asr.parts.k2.graph_transducer import GraphRnntLoss
    from nemo.core.utils.k2_guard import k2
except (ImportError, ModuleNotFoundError):
    pytest.skip("k2 is not installed, skipping Graph-RNNT tests.", allow_module_level=True)

EPS_SM_INPUT = 1e-6
EPS_L_INPUT = 1e-4

DEVICES = ['cpu']

if torch.cuda.is_available() and k2.with_cuda:
    DEVICES.append('cuda')


class TestGraphRnnt:
    @pytest.mark.unit
    def test_temporal_scheme(self):
        # TODO
        pass

    @pytest.mark.unit
    def test_unit_scheme(self):
        # TODO
        pass

    @pytest.mark.unit
    def test_grid_scheme(self):
        # TODO
        pass

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("connect_composed", [True, False])
    @pytest.mark.parametrize("blank_first", [True, False])
    def test_small_compose_transducer(
        self, device, connect_composed, blank_first, rnnt_test_helper, rnn_loss_sample_data
    ):
        if blank_first:
            sample_data = rnn_loss_sample_data.get_sample_small()
        else:
            sample_data = rnn_loss_sample_data.get_sample_small_blank_last()
        graph_rnnt = GraphRnntLoss(
            blank=sample_data.blank_id, connect_composed=connect_composed, use_grid_implementation=False
        )
        graph_cost, graph_grads = rnnt_test_helper.wrap_and_call(
            graph_rnnt, sample_data.logits, sample_data.targets, device
        )
        assert np.allclose(graph_cost, sample_data.expected_cost.numpy(), rtol=EPS_SM_INPUT), "costs mismatch."
        assert np.allclose(graph_grads, sample_data.expected_grads.numpy(), atol=1e-6), "gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_small_grid_transducer(self, device, rnnt_test_helper, rnn_loss_sample_data):
        sample_data = rnn_loss_sample_data.get_sample_small()
        graph_rnnt = GraphRnntLoss(blank=0, use_grid_implementation=True)
        graph_cost, graph_grads = rnnt_test_helper.wrap_and_call(
            graph_rnnt, sample_data.logits, sample_data.targets, device
        )
        assert np.allclose(graph_cost, sample_data.expected_cost.numpy(), rtol=EPS_SM_INPUT), "costs mismatch."
        assert np.allclose(graph_grads, sample_data.expected_grads.numpy(), atol=1e-6), "gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_medium_grid_transducer(self, device, rnnt_test_helper, rnn_loss_sample_data):
        sample_data = rnn_loss_sample_data.get_sample_medium()
        graph_rnnt = GraphRnntLoss(blank=0, use_grid_implementation=True)
        graph_cost, graph_grads = rnnt_test_helper.wrap_and_call(
            graph_rnnt, sample_data.logits, sample_data.targets, device
        )
        assert np.allclose(graph_cost, sample_data.expected_cost.numpy(), rtol=EPS_SM_INPUT), "costs mismatch."
        assert np.allclose(graph_grads, sample_data.expected_grads.numpy(), atol=1e-6), "gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_medium_random_var_size(self, device, rnnt_test_helper, rnn_loss_sample_data):
        sample_data = rnn_loss_sample_data.get_sample_medium_random_var_size(blank_first=True)
        graph_rnnt = GraphRnntLoss(blank=0, use_grid_implementation=True)
        graph_cost, graph_grads = rnnt_test_helper.wrap_and_call(
            graph_rnnt,
            sample_data.logits.detach(),
            sample_data.targets,
            device,
            input_lengths=sample_data.input_lengths,
            target_lengths=sample_data.target_lengths,
        )
        etalon_rnnt = RNNTLoss_Numpy(blank=0)
        etalon_cost, etalon_grads = rnnt_test_helper.wrap_and_call(
            etalon_rnnt,
            sample_data.logits.detach(),
            sample_data.targets,
            device,
            input_lengths=sample_data.input_lengths,
            target_lengths=sample_data.target_lengths,
        )
        assert np.allclose(graph_cost.sum(), etalon_cost, rtol=EPS_SM_INPUT), "costs mismatch."
        assert np.allclose(graph_grads, etalon_grads, atol=1e-4), "gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("blank_first", [True, False])
    def test_small_random_grid_compose_equivalent(self, device: torch.device, blank_first: bool, rnn_loss_sample_data):
        sample_data = rnn_loss_sample_data.get_sample_small_random(blank_first, device=device)
        criterion = GraphRnntLoss(blank=sample_data.blank_id, connect_composed=True, use_grid_implementation=False)
        text_tensor = sample_data.targets[0]
        sequence_length = sample_data.logits.shape[1]
        graph_grid = criterion.get_grid(text_tensor, sequence_length, sample_data.num_labels)
        graph_composed = criterion.get_composed_lattice(text_tensor, sequence_length, sample_data.num_labels)
        assert k2.is_rand_equivalent(
            graph_grid, graph_composed, log_semiring=True, treat_epsilons_specially=False
        ), "Grid and composed graphs are not equivalent."
