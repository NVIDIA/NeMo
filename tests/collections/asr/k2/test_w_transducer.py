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

import numpy as np
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
    @pytest.mark.parametrize("sequence_length", [1, 3, 6])
    @pytest.mark.parametrize("num_labels", [3])
    @pytest.mark.parametrize("last_blank_mode", ["force_last", "allow_ignore"])
    def test_temporal_scheme(self, device, blank_first, sequence_length, num_labels, last_blank_mode):
        blank_id = 0 if blank_first else num_labels - 1
        loss = GraphWTransducerLoss(blank=blank_id, last_blank_mode=last_blank_mode)
        temporal_scheme = loss.get_temporal_scheme(
            sequence_length=sequence_length, num_labels=num_labels, device=torch.device(device)
        )
        text_scheme: List[List[int]] = []
        for time_i in range(sequence_length):
            for label_i in range(num_labels):
                if label_i == blank_id:
                    # transition to the next state
                    text_scheme.append([time_i, time_i + 1, label_i, time_i, 0])
                else:
                    # self-loop
                    text_scheme.append([time_i, time_i, label_i, time_i, 0])

        # eps transitions from the first state
        eps_from_first_state = num_labels
        for time_i in range(1, sequence_length + 1):
            text_scheme.append([0, time_i, eps_from_first_state, 0, 0])

        # eps transitions to the last state
        eps_to_last_state = num_labels + 1
        last_state_eps = sequence_length - 1 if last_blank_mode == "force_last" else sequence_length
        for time_i in range(0, sequence_length - 1):
            text_scheme.append([time_i, last_state_eps, eps_to_last_state, time_i, 0])

        # transition to the final state
        text_scheme.append([sequence_length, sequence_length + 1, -1, -1, 0])
        # final state
        text_scheme.append([sequence_length + 1])

        text_scheme = sorted(text_scheme)  # required for k2.Fsa.from_str
        text_scheme_str = "\n".join([" ".join(map(str, line)) for line in text_scheme])
        etalon_temporal_scheme = k2.Fsa.from_str(text_scheme_str, num_aux_labels=1)
        assert temporal_scheme.num_arcs == etalon_temporal_scheme.num_arcs
        assert temporal_scheme.shape == etalon_temporal_scheme.shape  # (num_states, None)
        assert k2.is_rand_equivalent(
            temporal_scheme, etalon_temporal_scheme, log_semiring=True, treat_epsilons_specially=False
        ), "Temporal scheme mismatch"

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
    @pytest.mark.parametrize("blank_first", [True, False])
    @pytest.mark.parametrize("last_blank_mode", ["allow_ignore", "force_last"])
    def test_small_random_grid_compose_equivalent(
        self, device: torch.device, blank_first: bool, last_blank_mode, rnn_loss_sample_data
    ):
        sample_data = rnn_loss_sample_data.get_sample_small_random(blank_first, device=device)
        criterion = GraphWTransducerLoss(
            blank=sample_data.blank_id,
            last_blank_mode=last_blank_mode,
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

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("last_blank_mode", ["allow_ignore", "force_last"])
    @pytest.mark.parametrize("use_grid_implementation", [True, False])
    def test_small_grid_transducer_inf_penalty(
        self, device, last_blank_mode, use_grid_implementation, rnnt_test_helper, rnn_loss_sample_data
    ):
        """
        With -inf eps penalty W-Transducer loss should be equivalent to RNN-T loss.
        """
        sample_data = rnn_loss_sample_data.get_sample_small()
        graph_rnnt = GraphWTransducerLoss(
            blank=0,
            eps_weight=-100.0,
            last_blank_mode=last_blank_mode,
            use_grid_implementation=use_grid_implementation,
        )
        graph_cost, graph_grads = rnnt_test_helper.wrap_and_call(
            graph_rnnt, sample_data.logits, sample_data.targets, device
        )
        assert np.allclose(graph_cost, sample_data.expected_cost.numpy(), rtol=1e-6), "costs mismatch."
        assert np.allclose(graph_grads, sample_data.expected_grads.numpy(), atol=1e-6), "gradient mismatch."
