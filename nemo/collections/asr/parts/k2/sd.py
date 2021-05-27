# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright (c) 2020, Xiaomi CORPORATION.  All rights reserved.
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

from typing import Optional, Tuple, Union

import torch
import k2
import k2.sparse

from nemo.collections.asr.parts.k2.utils import create_supervision
from nemo.collections.asr.parts.k2.utils import get_tot_objf_and_num_frames
from nemo.collections.asr.parts.k2.utils import make_blank_first
from nemo.collections.asr.parts.k2.utils import load_graph


class SDLoss(torch.nn.Module):
    """
    Sequence-discriminative loss.
    Ported from https://github.com/k2-fsa/snowfall/blob/master/snowfall/objectives/mmi.py
    Implements Lattice-Free Maximum Mutual Information (LFMMI) 
    and Conditional Random Field based single-stage acoustic modeling (CRF) losses.
    """
    def __init__(
            self,
            num_classes: int,
            blank: int,
            reduction: str = 'mean',
            sd_type: str = 'mmi',
            tokel_lm: Optional[Union[k2.Fsa, str]] = None,
            tokel_lm_order: int = 2,
            den_scale: float = 1.0,
            calc_scores_pruned: bool = False,
            use_mbr: bool = False,
            decoding_graph: Optional[Union[k2.Fsa, str]] = None,
            **kwargs
    ):
        super().__init__()
        self.blank = blank
        self.num_classes = num_classes
        self.reduction = reduction
        self.sd_type = sd_type
        self.den_scale = den_scale
        self.use_mbr = use_mbr
        if not calc_scores_pruned and use_mbr:
            raise NotImplementedError("Not adapted yet")
        self.calc_scores = self._calc_scores_pruned if calc_scores_pruned else self._calc_scores_exact
        if tokel_lm is None:
            raise NotImplementedError("Not adapted yet")
        else:
            self.lm_graph = load_graph(tokel_lm) if isinstance(tokel_lm, str) else tokel_lm
        if sd_type == 'mmi':
            from nemo.collections.asr.parts.k2.graph_compilers import MmiTrainingGraphCompiler as compiler
        elif sd_type == 'crf':
            from nemo.collections.asr.parts.k2.graph_compilers import CtcCrfTrainingGraphCompiler as compiler
        else:
            raise ValueError(f"Invalid value of `sd_type`: {sd_type}.")
        self.graph_compiler = compiler(self.num_classes, aux_graph=self.lm_graph)
        if use_mbr:
            self.decoding_graph = load_graph(decoding_graph) if isinstance(decoding_graph, str) else decoding_graph
            if len(self.decoding_graph.shape) == 2:
                self.decoding_graph = k2.create_fsa_vec([self.decoding_graph])
        else:
            self.decoding_graph = None

    def _calc_scores_exact(self, dense_fsa_vec: k2.DenseFsaVec, num_graphs: k2.Fsa, den_graph: k2.Fsa, return_lats: bool = False):
        device = num_graphs.device

        num_fsas = num_graphs.shape[0]
        assert dense_fsa_vec.dim0() == num_fsas

        den_graph = den_graph.clone()
        num_graphs = num_graphs.clone()

        # The following converts aux_labels of den_graph and num_graphs
        # from torch.Tensor to k2.RaggedInt so that
        # we can use k2.append() later
        den_graph.convert_attr_to_ragged_(name='aux_labels')
        num_graphs.convert_attr_to_ragged_(name='aux_labels')

        num_den_graphs = k2.cat([num_graphs, den_graph])

        # NOTE: The a_to_b_map in k2.intersect_dense must be sorted
        # so the following reorders num_den_graphs.

        # [0, 1, 2, ... ]
        num_graphs_indexes = torch.arange(num_fsas, dtype=torch.int32)

        # [num_fsas, num_fsas, num_fsas, ... ]
        den_graph_indexes = torch.tensor([num_fsas] * num_fsas,
                                          dtype=torch.int32)

        # [0, num_fsas, 1, num_fsas, 2, num_fsas, ... ]
        num_den_graphs_indexes = torch.stack(
            [num_graphs_indexes, den_graph_indexes]).t().reshape(-1).to(device)

        num_den_reordered_graphs = k2.index(num_den_graphs, num_den_graphs_indexes)

        # [[0, 1, 2, ...]]
        a_to_b_map = torch.arange(num_fsas, dtype=torch.int32).reshape(1, -1)

        # [[0, 1, 2, ...]] -> [0, 0, 1, 1, 2, 2, ... ]
        a_to_b_map = a_to_b_map.repeat(2, 1).t().reshape(-1).to(device)

        num_den_lats = k2.intersect_dense(num_den_reordered_graphs,
                                          dense_fsa_vec,
                                          output_beam=10.0,
                                          a_to_b_map=a_to_b_map)

        num_den_tot_scores = num_den_lats.get_tot_scores(
            log_semiring=True, use_double_scores=False)

        num_tot_scores = num_den_tot_scores[::2]
        den_tot_scores = num_den_tot_scores[1::2]
        if return_lats:
            return num_tot_scores, den_tot_scores, num_den_lats
        else:
            return num_tot_scores, den_tot_scores

    def _calc_scores_pruned(self, dense_fsa_vec: k2.DenseFsaVec, num_graphs: k2.Fsa, den_graph: k2.Fsa, return_lats: bool = False):
        num_fsas = num_graphs.shape[0]
        assert dense_fsa_vec.dim0() == num_fsas

        # indexes = torch.zeros(num_fsas, dtype=torch.int32, device=den_graph.device)
        # den_graphs = k2.index_fsa(den_graph, indexes).to(den_graph.device)
        den_graphs = den_graph

        num_lats = k2.intersect_dense(num_graphs,
                                      dense_fsa_vec,
                                      output_beam=torch.finfo(torch.float32).max,
                                      seqframe_idx_name='seqframe_idx')
        # den_lats = k2.intersect_dense(den_graphs, dense_fsa_vec, 10.0)
        den_lats = k2.intersect_dense_pruned(den_graphs,
                                             dense_fsa_vec,
                                             search_beam=20.0,
                                             output_beam=7.0,
                                             min_active_states=30,
                                             max_active_states=10000,
                                             seqframe_idx_name='seqframe_idx')

        num_tot_scores = num_lats.get_tot_scores(
            log_semiring=True,
            use_double_scores=False
        )
        den_tot_scores = den_lats.get_tot_scores(
            log_semiring=True,
            use_double_scores=False
        )
        if return_lats:
            return num_tot_scores, den_tot_scores, num_lats, den_lats
        else:
            return num_tot_scores, den_tot_scores

    def forward(
            self,
            log_probs: torch.Tensor,
            targets: torch.Tensor,
            input_lengths: torch.Tensor,
            target_lengths: torch.Tensor
    ) -> torch.Tensor:
        if self.blank != 0:
            # rearrange log_probs to put blank at the first place
            # and shift targets to emulate blank = 0
            log_probs, targets = make_blank_first(self.blank, log_probs, targets)
        supervisions, order = create_supervision(input_lengths)
        targets = targets[order]
        target_lengths = target_lengths[order]

        if log_probs.device != self.graph_compiler.device:
            self.graph_compiler.to(log_probs.device)
            self.lm_graph = self.lm_graph.to(log_probs.device)
        if self.use_mbr and log_probs.device != self.decoding_graph.device:
            self.decoding_graph = self.decoding_graph.to(log_probs.device)

        num_graphs, den_graph = self.graph_compiler.compile(targets, target_lengths)

        dense_fsa_vec = k2.DenseFsaVec(log_probs, supervisions)

        if self.use_mbr:
            den_graph = self.decoding_graph
        calc_scores_result = self.calc_scores(dense_fsa_vec, num_graphs, den_graph, self.use_mbr)
        num_tot_scores, den_tot_scores = calc_scores_result[:2]
        if self.sd_type == 'crf':
            token_ids_list = [t[:l].tolist() for t, l in zip(targets, target_lengths)]
            label_graph = k2.linear_fsa(token_ids_list).to(num_tot_scores.device)
            path_weight_graphs = k2.compose(self.lm_graph, label_graph, treat_epsilons_specially=False)
            path_weight_graphs = k2.arc_sort(path_weight_graphs)
            num_tot_scores += path_weight_graphs._get_tot_scores(False, True)
        tot_scores = num_tot_scores - self.den_scale * den_tot_scores
        tot_scores, _, _ = get_tot_objf_and_num_frames(
            tot_scores,
            supervisions[:, 2],
            self.reduction
        )

        if self.use_mbr:
            num_lats, den_lats = calc_scores_result[2:]
            num_rows = dense_fsa_vec.scores.shape[0]
            num_cols = dense_fsa_vec.scores.shape[1] - 1
            mbr_num_sparse = k2.create_sparse(rows=num_lats.seqframe_idx,
                                              cols=num_lats.phones,
                                              values=num_lats.get_arc_post(False,
                                                                           True).exp(),
                                              size=(num_rows, num_cols),
                                              min_col_index=0)

            mbr_den_sparse = k2.create_sparse(rows=den_lats.seqframe_idx,
                                              cols=den_lats.phones,
                                              values=den_lats.get_arc_post(False,
                                                                           True).exp(),
                                              size=(num_rows, num_cols),
                                              min_col_index=0)
            # NOTE: Due to limited support of PyTorch's autograd for sparse tensors,
            # we cannot use (mbr_num_sparse - mbr_den_sparse) here
            #
            # The following works only for torch >= 1.7.0
            mbr_loss = torch.sparse.sum(
                k2.sparse.abs((mbr_num_sparse + (-mbr_den_sparse)).coalesce()))
            total_loss = mbr_loss - tot_scores
        else:
            total_loss = - tot_scores
        return total_loss
