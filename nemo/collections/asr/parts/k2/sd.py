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

from typing import Optional, Union

import torch
import k2
import k2.sparse
import _k2

from nemo.collections.asr.parts.k2.autograd import sparse_abs
from nemo.collections.asr.parts.k2.utils import create_supervision
from nemo.collections.asr.parts.k2.utils import create_sparse_wrapped
from nemo.collections.asr.parts.k2.utils import get_tot_objf_and_num_frames
from nemo.collections.asr.parts.k2.utils import make_blank_first
from nemo.collections.asr.parts.k2.utils import load_graph
from nemo.collections.asr.parts.k2.utils import prep_padded_densefsavec
from nemo.collections.asr.parts.k2.utils import shift_labels_inpl
from nemo.collections.asr.parts.k2.utils import GradExpNormalize
from nemo.collections.asr.parts.k2.utils import GradScale
from nemo.utils import logging


class SDLoss(torch.nn.Module):
    """
    Sequence-discriminative loss.
    Ported from https://github.com/k2-fsa/snowfall/blob/master/snowfall/objectives/mmi.py
    Implements Lattice-Free Maximum Mutual Information (LF-MMI) 
    and Conditional Random Field based single-stage acoustic modeling (CRF) losses.
    """
    def __init__(
            self,
            num_classes: int,
            blank: int,
            reduction: str = 'mean',
            topo_type: str = 'ctc_default',
            topo_with_selfloops: bool = True,
            sd_type: str = 'mmi',
            token_lm: Optional[Union[k2.Fsa, str]] = None,
            token_lm_order: int = 2,
            den_scale: float = 1.0,
            intersect_pruned: bool = False,
            use_mbr: bool = False,
            mbr_graph: Optional[Union[k2.Fsa, str]] = None,
            **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.blank = blank
        self.reduction = reduction
        self.sd_type = sd_type
        self.den_scale = den_scale
        self.use_mbr = use_mbr
        self.intersect_mmi = self._intersect_mmi_pruned if intersect_pruned else self._intersect_mmi_exact
        self.pad_fsavec = topo_type == "ctc_compact"
        if token_lm is None:
            logging.warning(f"""token_lm is empty. 
                            Using loss without token_lm will result in error.""")
            self.lm_graph = token_lm
        else:
            self.lm_graph = load_graph(token_lm) if isinstance(token_lm, str) else token_lm
            if hasattr(self.lm_graph, 'aux_labels'):
                delattr(self.lm_graph, 'aux_labels')
            if self.pad_fsavec:
                shift_labels_inpl([self.lm_graph], 1)
        if sd_type == 'mmi':
            from nemo.collections.asr.parts.k2.graph_compilers import MmiTrainingGraphCompiler as compiler
        elif sd_type == 'crf':
            from nemo.collections.asr.parts.k2.graph_compilers import CtcCrfTrainingGraphCompiler as compiler
        else:
            raise ValueError(f"Invalid value of `sd_type`: {sd_type}.")
        self.graph_compiler = compiler(self.num_classes, topo_type, topo_with_selfloops, aux_graph=self.lm_graph)
        if use_mbr and mbr_graph is not None:
            self.mbr_graph = load_graph(mbr_graph) if isinstance(mbr_graph, str) else mbr_graph
            if len(self.mbr_graph.shape) == 2:
                self.mbr_graph = k2.create_fsa_vec([self.mbr_graph])
        else:
            self.mbr_graph = None

    def _intersect_mmi_exact(self, dense_fsa_vec: k2.DenseFsaVec, num_graphs: k2.Fsa, den_graph: k2.Fsa):
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
        den_graph_indexes = torch.tensor([num_fsas] * num_fsas, dtype=torch.int32)

        # [0, num_fsas, 1, num_fsas, 2, num_fsas, ... ]
        num_den_graphs_indexes = torch.stack(
            [num_graphs_indexes, den_graph_indexes]).t().reshape(-1).to(device)

        num_den_reordered_graphs = k2.index(num_den_graphs, num_den_graphs_indexes)

        # [[0, 1, 2, ...]]
        a_to_b_map = torch.arange(num_fsas, dtype=torch.int32).reshape(1, -1)

        # [[0, 1, 2, ...]] -> [0, 0, 1, 1, 2, 2, ... ]
        a_to_b_map = a_to_b_map.repeat(2, 1).t().reshape(-1).to(device)

        num_den_lats = k2.intersect_dense(a_fsas=num_den_reordered_graphs,
                                          b_fsas=dense_fsa_vec,
                                          output_beam=10.0,
                                          a_to_b_map=a_to_b_map,
                                          seqframe_idx_name='seqframe_idx')
        lat_slice = torch.arange(num_fsas, dtype=torch.int32).to(device) * 2
        return k2.index(num_den_lats, lat_slice), k2.index(num_den_lats, lat_slice + 1)

    def _intersect_mmi_pruned(self, dense_fsa_vec: k2.DenseFsaVec, num_graphs: k2.Fsa, den_graph: k2.Fsa):
        num_fsas = num_graphs.shape[0]
        assert dense_fsa_vec.dim0() == num_fsas

        num_lats = k2.intersect_dense(a_fsas=num_graphs,
                                      b_fsas=dense_fsa_vec,
                                      output_beam=torch.finfo(torch.float32).max,
                                      seqframe_idx_name='seqframe_idx')
        den_lats = k2.intersect_dense_pruned(a_fsas=den_graph,
                                             b_fsas=dense_fsa_vec,
                                             search_beam=20.0,
                                             output_beam=7.0,
                                             min_active_states=30,
                                             max_active_states=10000,
                                             seqframe_idx_name='seqframe_idx')
        return num_lats, den_lats

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
        if self.sd_type == 'crf':
            # Same as for CTC
            log_probs = GradExpNormalize.apply(log_probs, input_lengths, "mean" if self.reduction != "sum" else "none")

        if log_probs.device != self.graph_compiler.device:
            self.graph_compiler.to(log_probs.device)
            self.lm_graph = self.lm_graph.to(log_probs.device)
        if self.use_mbr and self.mbr_graph is not None and log_probs.device != self.mbr_graph.device:
            self.mbr_graph = self.mbr_graph.to(log_probs.device)

        num_graphs, den_graph = self.graph_compiler.compile(targets + 1 if self.pad_fsavec else targets, target_lengths)

        if self.use_mbr and self.mbr_graph is not None:
            den_graph = self.mbr_graph

        dense_fsa_vec = prep_padded_densefsavec(log_probs, supervisions) if self.pad_fsavec else k2.DenseFsaVec(log_probs, supervisions)

        num_lats, den_lats = self.intersect_mmi(dense_fsa_vec, num_graphs, den_graph)
        # if self.pad_fsavec:
            # shift_labels_inpl([num_lats, den_lats], -1)

        # use_double_scores=True does matter
        # since otherwise it sometimes makes rounding errors
        num_tot_scores = num_lats.get_tot_scores(
            log_semiring=True,
            use_double_scores=True
        )
        den_tot_scores = den_lats.get_tot_scores(
            log_semiring=True,
            use_double_scores=True
        )

        if self.sd_type == 'crf':
            token_ids_list = [t[:l].tolist() for t, l in zip(targets, target_lengths)]
            label_graph = k2.linear_fsa(token_ids_list).to(num_tot_scores.device)
            path_weight_graphs = k2.compose(self.lm_graph, label_graph, treat_epsilons_specially=False)
            path_weight_graphs = k2.arc_sort(path_weight_graphs)
            num_tot_scores += path_weight_graphs._get_tot_scores(False, True)
        # tot_scores = num_tot_scores - self.den_scale * den_tot_scores
        # alaptev: I believe it is better to vary only gradients for the sake of comparability
        tot_scores = num_tot_scores - GradScale.apply(den_tot_scores, self.den_scale)
        mmi_tot_scores, mmi_valid_mask = get_tot_objf_and_num_frames(
            tot_scores,
            self.reduction
        )

        # In crf training den_tot_scores can exceed num_tot_scores.
        # It means that the model is going to diverge.
        # It can also be triggered when switching the lm_graph, so I commented it out for now.
        # assert tot_scores.nelement() == 0 or torch.all(tot_scores <= 0.0), "denominator took over"

        if self.use_mbr:
            size = (dense_fsa_vec.dim0(), dense_fsa_vec.scores.shape[0], dense_fsa_vec.scores.shape[1] - 1)
            row_ids = dense_fsa_vec.dense_fsa_vec.shape().row_ids(1)
            mbr_num_sparse = create_sparse_wrapped(indices=[_k2.index_select(row_ids, num_lats.seqframe_idx),
                                                            num_lats.seqframe_idx,
                                                            num_lats.phones],
                                                   values=num_lats.get_arc_post(True,True).exp(),
                                                   size=size,
                                                   min_col_index=0)
            mbr_den_sparse = create_sparse_wrapped(indices=[_k2.index_select(row_ids, den_lats.seqframe_idx),
                                                            den_lats.seqframe_idx,
                                                            den_lats.phones],
                                                   values=den_lats.get_arc_post(True,True).exp(),
                                                   size=size,
                                                   min_col_index=0)
            # NOTE: Due to limited support of PyTorch's autograd for sparse tensors,
            # we cannot use (mbr_num_sparse - mbr_den_sparse) here
            # The following works only for torch >= 1.7.0
            mbr_loss = torch.sparse.sum(sparse_abs((mbr_num_sparse + (-mbr_den_sparse)).coalesce()), (1,2)).to_dense()
            mbr_tot_scores, mbr_valid_mask = get_tot_objf_and_num_frames(
                mbr_loss,
                self.reduction
            )
            valid_mask = mmi_valid_mask & mbr_valid_mask
            total_loss = mbr_tot_scores[valid_mask] - mmi_tot_scores[valid_mask]
        else:
            total_loss = - mmi_tot_scores[mmi_valid_mask]
        return total_loss
