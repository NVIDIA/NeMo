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

import math
from contextlib import nullcontext
from typing import Union

import torch
import torch.nn.functional as F

from nemo.collections.asr.parts.k2.graph_transducer import GraphRnntLoss, force_float32_context
from nemo.core.utils.k2_guard import k2
from nemo.utils.enum import PrettyStrEnum


class GraphWTransducerLoss(GraphRnntLoss):
    class LastBlankMode(PrettyStrEnum):
        ALLOW_IGNORE = "allow_ignore"
        FORCE_LAST = "force_last"
        FORCE_ALL = "force_all"

    class GraphMode(PrettyStrEnum):
        SEQUENTIAL = "sequential"
        SKIP_FRAMES = "skip_frames"

    def __init__(
        self,
        blank: int,
        eps_weight=math.log(0.5),
        last_blank_mode: Union[LastBlankMode, str] = LastBlankMode.FORCE_LAST,
        graph_mode: Union[GraphMode, str] = GraphMode.SKIP_FRAMES,
        use_grid_implementation=True,
        connect_composed=False,
        double_scores=False,
        cast_to_float32=False,
    ):
        super().__init__(
            blank=blank,
            use_grid_implementation=use_grid_implementation,
            connect_composed=connect_composed,
            double_scores=double_scores,
            cast_to_float32=cast_to_float32,
        )
        self._eps_weight = eps_weight
        self._last_blank_mode = self.LastBlankMode(last_blank_mode)
        self._graph_mode = self.GraphMode(graph_mode)

    def get_eps_id(self, num_labels: int) -> int:
        return num_labels

    def get_unit_scheme(self, text_tensor: torch.Tensor, num_labels: int) -> "k2.Fsa":
        blank_id = self.blank
        start_eps_id = num_labels
        end_eps_id = num_labels + 1
        device = text_tensor.device
        text_len = text_tensor.shape[0]

        # arcs: scr, dest, label, score
        arcs = torch.zeros(((text_len + 1) * 2 + 2, 4), dtype=torch.int32, device=device)
        text_indices = torch.arange(0, text_len + 1, dtype=torch.int32, device=device)
        # eps
        arcs[0, 2] = start_eps_id
        # blank labels
        arcs[1:-1:2, 0] = text_indices  # from state
        arcs[1:-1:2, 1] = text_indices  # to state
        arcs[1:-1:2, 2] = blank_id

        # text labels
        arcs[2:-1:2, 0] = text_indices  # from state
        arcs[2:-1:2, 1] = text_indices + 1  # to state
        arcs[2:-2:2, 2] = text_tensor  # labels: text

        arcs[-1] = arcs[-2]
        arcs[-2, 1] = text_len
        arcs[-2, 2] = end_eps_id
        arcs[-1, 2] = -1  # last transition to final state, ilabel=-1 (special for k2)
        olabels = arcs[:, 2].detach().clone()  # same as ilabels

        fsa_text = k2.Fsa(arcs, olabels)
        fsa_text.text_positions = torch.zeros_like(olabels)
        fsa_text.text_positions[1:-1] = text_indices.expand(2, -1).transpose(0, 1).flatten()
        fsa_text.text_positions[-1] = fsa_text.text_positions[-2]
        return fsa_text

    def get_temporal_scheme(self, sequence_length: int, num_labels: int, device: torch.device) -> "k2.Fsa":
        blank_id = self.blank
        start_eps_id = num_labels
        end_eps_id = num_labels + 1
        num_eps = 2
        last_eps_ark = (
            self._graph_mode == self.GraphMode.SEQUENTIAL and self._last_blank_mode == self.LastBlankMode.ALLOW_IGNORE
        )

        num_sequence_arcs = sequence_length * (num_labels + num_eps)
        fsa_temporal_arcs = torch.zeros((num_sequence_arcs + int(last_eps_ark), 4), dtype=torch.int32, device=device)
        sequence_states = torch.arange(0, sequence_length, dtype=torch.int32, device=device)
        sequence_states_next = sequence_states + 1
        # for every state - num_labels+1 arcs, [0, 1, ..., num_labels-1, eps, 0, 1, ..., num_labels-1, eps, ...]
        start_states = sequence_states.expand(num_labels + num_eps, sequence_length).transpose(0, 1).flatten()

        # self-loops - all, make forward arcs later
        fsa_temporal_arcs[:num_sequence_arcs, 0] = start_states  # from
        fsa_temporal_arcs[:num_sequence_arcs, 1] = start_states  # to
        fsa_temporal_arcs[:num_sequence_arcs, 2] = (
            torch.arange(0, num_labels + num_eps, dtype=torch.int32, device=device)
            .expand(sequence_length, num_labels + num_eps)
            .flatten()
        )
        # forward arcs
        fsa_temporal_arcs[blank_id : num_sequence_arcs : num_labels + num_eps, 1] = sequence_states_next  # blanks
        # eps arcs
        if self._graph_mode == self.GraphMode.SEQUENTIAL:
            fsa_temporal_arcs[start_eps_id : num_sequence_arcs : num_labels + num_eps, 1] = sequence_states_next
            fsa_temporal_arcs[end_eps_id : num_sequence_arcs : num_labels + num_eps, 1] = sequence_states_next
        else:
            # self._graph_mode == self.GraphMode.SKIP_FRAMES
            fsa_temporal_arcs[start_eps_id : num_sequence_arcs : num_labels + num_eps, 0] = 0
            fsa_temporal_arcs[start_eps_id : num_sequence_arcs : num_labels + num_eps, 1] = sequence_states + 1
            fsa_temporal_arcs[end_eps_id : num_sequence_arcs : num_labels + num_eps, 0] = sequence_states
            fsa_temporal_arcs[end_eps_id : num_sequence_arcs : num_labels + num_eps, 1] = (
                sequence_length - 1 if self._last_blank_mode == self.LastBlankMode.FORCE_LAST else sequence_length
            )

        # transition to last final state
        fsa_temporal_arcs[-1, :3] = torch.tensor(
            (sequence_length, sequence_length + 1, -1), dtype=torch.int32, device=device
        )

        if self._graph_mode == self.GraphMode.SKIP_FRAMES:
            # need to sort arcs
            _, indices = torch.sort(fsa_temporal_arcs[:, 0], dim=0)
            fsa_temporal_arcs = fsa_temporal_arcs[indices]

        # output symbols: position in the sequence, same as start states for arcs
        olabels = fsa_temporal_arcs[:, 0].detach().clone()

        fsa_temporal = k2.Fsa(fsa_temporal_arcs, olabels)
        fsa_temporal = k2.arc_sort(fsa_temporal)  # need for compose
        return fsa_temporal

    def get_grid(self, text_tensor: torch.Tensor, sequence_length: int, num_labels: int) -> "k2.Fsa":
        blank_id = self.blank
        eps_id = self.get_eps_id(num_labels)
        text_length = text_tensor.shape[0]
        device = text_tensor.device
        num_grid_states = sequence_length * (text_length + 1)
        last_eps_ark = (
            self._last_blank_mode == self.LastBlankMode.ALLOW_IGNORE and self._graph_mode == self.GraphMode.SEQUENTIAL
        )
        num_forward_arcs_base = (sequence_length - 1) * (text_length + 1)
        num_forward_arcs_additional = (sequence_length - 1) * 2 + int(last_eps_ark)
        num_forward_arcs = num_forward_arcs_base + num_forward_arcs_additional
        num_text_arcs = text_length * sequence_length
        arcs = torch.zeros((num_forward_arcs + num_text_arcs + 2, 4), dtype=torch.int32, device=device)
        # blank transitions
        # i, i+<text_len + 1>, 0 <blank>, i / <text_len+1>, i % <text_len + 1>
        from_states = torch.arange(num_forward_arcs_base, device=device)
        to_states = from_states + (text_length + 1)
        arcs[:num_forward_arcs_base, 0] = from_states
        arcs[:num_forward_arcs_base, 1] = to_states
        arcs[:num_forward_arcs_base, 2] = blank_id

        from_states = torch.cat(
            [
                torch.arange(sequence_length - 1, device=device) * (text_length + 1),
                text_length + torch.arange(sequence_length - 1, device=device) * (text_length + 1),
            ]
        )
        to_states = from_states + (text_length + 1)
        arcs[num_forward_arcs_base : num_forward_arcs_base + (sequence_length - 1) * 2, 0] = from_states
        arcs[num_forward_arcs_base : num_forward_arcs_base + (sequence_length - 1) * 2, 1] = to_states
        arcs[num_forward_arcs_base : num_forward_arcs_base + (sequence_length - 1), 2] = eps_id
        arcs[num_forward_arcs_base + (sequence_length - 1) : num_forward_arcs_base + (sequence_length - 1) * 2, 2] = (
            eps_id + 1
        )
        if self._graph_mode == self.GraphMode.SKIP_FRAMES:
            arcs[num_forward_arcs_base : num_forward_arcs_base + (sequence_length - 1), 0] = 0
            arcs[
                num_forward_arcs_base + (sequence_length - 1) : num_forward_arcs_base + (sequence_length - 1) * 2, 1
            ] = (
                num_grid_states - 1
            )  # if other mode - fix later
        # last eps ark - after relabel

        # text arcs
        from_states = (
            torch.arange(num_grid_states, dtype=torch.int32, device=device)
            .reshape(sequence_length, text_length + 1)[:, :-1]
            .flatten()
        )
        to_states = from_states + 1
        ilabels = text_tensor.expand(sequence_length, -1).flatten()
        arcs[num_forward_arcs:-2, 0] = from_states
        arcs[num_forward_arcs:-2, 1] = to_states
        arcs[num_forward_arcs:-2, 2] = ilabels

        # last 2 states
        arcs[-2, :3] = torch.tensor((num_grid_states - 1, num_grid_states, blank_id), dtype=torch.int32, device=device)
        arcs[-1, :3] = torch.tensor((num_grid_states, num_grid_states + 1, -1), dtype=torch.int32, device=device)

        # sequence indices, time indices
        olabels = torch.div(arcs[:, 0], (text_length + 1), rounding_mode="floor")  # arcs[:, 0] // (text_length + 1)
        text_positions = arcs[:, 0] % (text_length + 1)
        # last state: final
        olabels[-1] = -1
        text_positions[-1] = -1

        # relabel
        # instead of using top sort (extremely expensive) k2.top_sort(rnnt_graph)
        arcs[:-2, 0] = self.relabel_states(arcs[:-2, 0], text_length + 1, sequence_length)
        arcs[:-3, 1] = self.relabel_states(arcs[:-3, 1], text_length + 1, sequence_length)

        if last_eps_ark:
            arcs[num_forward_arcs - 1, 0] = num_grid_states - 1
            arcs[num_forward_arcs - 1, 1] = num_grid_states
            arcs[num_forward_arcs - 1, 2] = eps_id + 1
        if (
            self._last_blank_mode == self.LastBlankMode.ALLOW_IGNORE
            or self._last_blank_mode == self.LastBlankMode.FORCE_ALL
        ) and self._graph_mode == self.GraphMode.SKIP_FRAMES:
            arcs[
                num_forward_arcs_base + (sequence_length - 1) : num_forward_arcs_base + (sequence_length - 1) * 2, 1
            ] = num_grid_states

        # sort by start state - required in k2
        # TODO: maybe it is more optimal to avoid sort, construct arcs in ascending order
        _, indices = torch.sort(arcs[:, 0], dim=0)
        arcs = arcs[indices]
        olabels = olabels[indices]
        text_positions = text_positions[indices]
        if self._last_blank_mode == self.LastBlankMode.FORCE_ALL:
            arcs[arcs[:, 2] == eps_id + 1, 2] = blank_id

        rnnt_graph = k2.Fsa(arcs, olabels)
        rnnt_graph.text_positions = text_positions
        return rnnt_graph

    def forward(
        self, acts: torch.Tensor, labels: torch.Tensor, act_lens: torch.Tensor, label_lens: torch.Tensor,
    ):
        # nemo: acts=log_probs, labels=targets, act_lens=input_lengths, label_lens=target_lengths
        logits, targets, logits_lengths, target_lengths = acts, labels, act_lens, label_lens

        # logits: B x Time x Text+1 x C
        num_labels = logits.shape[-1]
        target_fsas_vec = self.get_graphs_batched(logits_lengths, targets, target_lengths, num_labels)

        cast_context = force_float32_context() if self.cast_to_float32 else nullcontext()
        with cast_context:
            log_probs = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                indices = self.get_logits_indices(target_fsas_vec, logits.shape)
                indices[target_fsas_vec.labels == -1] = 0
                indices[target_fsas_vec.labels >= num_labels] = 0  # eps

            # NB: do not assign scores -> modify, k2 will not update all scores correctly (modify -> assign)
            scores = log_probs.flatten().index_select(-1, indices)
            scores[target_fsas_vec.labels == -1] = 0
            scores[target_fsas_vec.labels >= num_labels] = self._eps_weight  # eps

            target_fsas_vec.scores = scores
            scores = -1 * target_fsas_vec.get_tot_scores(use_double_scores=self._double_scores, log_semiring=True)
            return scores
