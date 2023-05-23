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

        etalon_scheme_fst: List[List[int]] = []
        for time_i in range(sequence_length):
            for label_i in range(num_labels):
                if label_i == blank_id:
                    # transition to the next state
                    etalon_scheme_fst.append([time_i, time_i + 1, label_i, time_i, 0])
                else:
                    # self-loop
                    etalon_scheme_fst.append([time_i, time_i, label_i, time_i, 0])

        # eps transitions from the first state
        eps_from_first_state = num_labels
        for time_i in range(1, sequence_length + 1):
            etalon_scheme_fst.append([0, time_i, eps_from_first_state, 0, 0])

        # eps transitions to the last state
        eps_to_last_state = num_labels + 1
        last_state_eps = sequence_length - 1 if last_blank_mode == "force_last" else sequence_length
        for time_i in range(0, sequence_length - 1):
            etalon_scheme_fst.append([time_i, last_state_eps, eps_to_last_state, time_i, 0])

        # transition to the final state
        etalon_scheme_fst.append([sequence_length, sequence_length + 1, -1, -1, 0])
        # final state
        etalon_scheme_fst.append([sequence_length + 1])

        etalon_scheme_fst = sorted(etalon_scheme_fst)  # required for k2.Fsa.from_str
        etalon_scheme_fst_str = "\n".join([" ".join(map(str, line)) for line in etalon_scheme_fst])
        etalon_temporal_scheme = k2.Fsa.from_str(etalon_scheme_fst_str, num_aux_labels=1)

        assert temporal_scheme.num_arcs == etalon_temporal_scheme.num_arcs
        assert temporal_scheme.shape == etalon_temporal_scheme.shape  # (num_states, None)
        assert k2.is_rand_equivalent(
            temporal_scheme, etalon_temporal_scheme, log_semiring=True, treat_epsilons_specially=False
        ), "Temporal scheme mismatch"
        assert k2.is_rand_equivalent(
            temporal_scheme.invert(),
            etalon_temporal_scheme.invert(),
            log_semiring=False,
            treat_epsilons_specially=False,
        ), "Temporal scheme output labels mismatch"

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("blank_first", [True, False])
    def test_unit_scheme(self, device, blank_first):
        num_labels = 3
        blank_id = 0 if blank_first else num_labels - 1
        if blank_first:
            labels = [1, 1, 2, 1]
        else:
            labels = [1, 1, 0, 1]
        loss = GraphWTransducerLoss(blank=blank_id)
        unit_scheme = loss.get_unit_scheme(
            units_tensor=torch.tensor(labels, device=torch.device(device)), num_labels=num_labels
        )

        etalon_scheme_fst: List[List[int]] = []
        for label_i, label in enumerate(labels):
            etalon_scheme_fst.append([label_i, label_i + 1, label, label, label_i, 0])  # forward: label
            etalon_scheme_fst.append([label_i, label_i, blank_id, blank_id, label_i, 0])  # self-loop: blank
        etalon_scheme_fst.append([len(labels), len(labels), blank_id, blank_id, len(labels), 0])
        # eps-transitions
        etalon_scheme_fst.append([0, 0, num_labels, num_labels, 0, 0])
        etalon_scheme_fst.append([len(labels), len(labels), num_labels + 1, num_labels + 1, len(labels), 0])

        etalon_scheme_fst.append([len(labels), len(labels) + 1, -1, -1, -1, 0])  # transition to final state
        etalon_scheme_fst.append([len(labels) + 1])  # final state
        etalon_scheme_fst = sorted(etalon_scheme_fst)  # required for k2.Fsa.from_str
        etalon_scheme_fst_str = "\n".join([" ".join(map(str, line)) for line in etalon_scheme_fst])
        etalon_unit_scheme = k2.Fsa.from_str(etalon_scheme_fst_str, aux_label_names=["aux_labels", "unit_positions"])

        assert unit_scheme.num_arcs == etalon_unit_scheme.num_arcs
        assert unit_scheme.shape == etalon_unit_scheme.shape  # (num_states, None)
        assert k2.is_rand_equivalent(
            unit_scheme, etalon_unit_scheme, log_semiring=True, treat_epsilons_specially=False
        ), "Unit scheme input labels mismatch"
        assert k2.is_rand_equivalent(
            unit_scheme.invert(), etalon_unit_scheme.invert(), log_semiring=True, treat_epsilons_specially=False
        ), "Unit scheme output labels mismatch"

        # swap aux_labels and unit positions to test unit_positions
        unit_scheme.aux_labels, unit_scheme.unit_positions = unit_scheme.unit_positions, unit_scheme.aux_labels
        etalon_unit_scheme.aux_labels, etalon_unit_scheme.unit_positions = (
            etalon_unit_scheme.unit_positions,
            etalon_unit_scheme.aux_labels,
        )
        assert k2.is_rand_equivalent(
            unit_scheme.invert(), etalon_unit_scheme.invert(), log_semiring=True, treat_epsilons_specially=False
        ), "Unit scheme unit positions mismatch"

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
