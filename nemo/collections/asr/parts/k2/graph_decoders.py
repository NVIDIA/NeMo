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

from typing import List, Optional, Tuple, Union

import torch
import k2

from nemo.collections.asr.parts.k2.topologies import build_topo
from nemo.collections.asr.parts.k2.utils import create_supervision
from nemo.collections.asr.parts.k2.utils import intersect_with_self_loops
from nemo.collections.asr.parts.k2.utils import invert_permutation
from nemo.collections.asr.parts.k2.utils import make_blank_first
from nemo.collections.asr.parts.k2.utils import load_graph
from nemo.collections.asr.parts.k2.utils import prep_padded_densefsavec
from nemo.collections.asr.parts.k2.utils import shift_labels_inpl
from nemo.utils import logging


class DecoderConf:
    """TBD
    """
    def __init__(self,
                 search_beam=20.0,
                 output_beam=10.0,
                 min_active_states=30,
                 max_active_states=10000,
                 **kwargs
    ):
        self.search_beam = search_beam
        self.output_beam = output_beam
        self.min_active_states = min_active_states
        self.max_active_states = max_active_states


class BaseDecoder(object):
    """TBD
    """
    def __init__(self,
                 num_classes: int,
                 blank: int,
                 intersect_pruned: bool = False,
                 topo_type: str = "default",
                 topo_with_selfloops: bool = True,
                 device: torch.device = torch.device("cpu"),
                 **kwargs
    ):
        self.num_classes = num_classes
        self.blank = blank
        self.intersect_pruned = intersect_pruned
        self.device = device
        self.topo_type = topo_type
        self.ctc_topo_inv = k2.arc_sort(build_topo(topo_type, list(range(num_classes)), topo_with_selfloops).invert_())
        self.decode_graph = None
        self.pad_fsavec = topo_type == "ctc_compact"
        self.conf = DecoderConf(**kwargs)

    def to(self, device: torch.device):
        if self.decode_graph is not None:
            self.decode_graph = self.decode_graph.to(device)
        self.device = device

    def update_graph(self, graph: k2.Fsa):
        raise NotImplementedError

    def decode(self,
               log_probs: torch.Tensor,
               input_lengths: torch.Tensor,
               return_lattices: bool = False,
               return_ilabels: bool = False
    ) -> Union[k2.Fsa, Tuple[List[torch.Tensor], torch.Tensor]]:
        assert self.decode_graph is not None

        if self.blank != 0:
            # rearrange log_probs to put blank at the first place
            # and shift targets to emulate blank = 0
            log_probs, _ = make_blank_first(self.blank, log_probs, None)
        supervisions, order = create_supervision(input_lengths)

        if log_probs.device != self.device:
            self.to(log_probs.device)
        dense_fsa_vec = prep_padded_densefsavec(log_probs, supervisions) if self.pad_fsavec else k2.DenseFsaVec(log_probs, supervisions)

        if self.intersect_pruned:
            lats = k2.intersect_dense_pruned(a_fsas=self.decode_graph,
                                             b_fsas=dense_fsa_vec,
                                             search_beam=self.conf.search_beam,
                                             output_beam=self.conf.output_beam,
                                             min_active_states=self.conf.min_active_states,
                                             max_active_states=self.conf.max_active_states)
        else:
            indices = torch.zeros(dense_fsa_vec.dim0(), dtype=torch.int32, device=self.device)
            dec_graphs = k2.index_fsa(self.decode_graph, indices)
            lats = k2.intersect_dense(dec_graphs, dense_fsa_vec, self.conf.output_beam)
        if self.pad_fsavec:
            shift_labels_inpl([lats], -1)

        if return_lattices:
            lats = k2.index_fsa(lats, invert_permutation(order).to(dtype=torch.int32, device=input_lengths.device))
            if self.blank != 0:
                # change only ilabels
                # suppose self.blank == self.num_classes - 1
                lats.labels = torch.where(lats.labels == 0, self.blank, lats.labels - 1)
            return lats
        else:
            shortest_paths_fsa = k2.shortest_path(lats, True)
            shortest_paths_fsa = k2.index_fsa(shortest_paths_fsa, invert_permutation(order).to(dtype=torch.int32, device=input_lengths.device))
            scores = shortest_paths_fsa._get_tot_scores(True, False)
            if return_ilabels:
                shortest_paths = []
                # direct iterating does not work as expected
                for i in range(shortest_paths_fsa.shape[0]):
                    labels = shortest_paths_fsa[i].labels[:-1].to(dtype=torch.long)
                    if self.blank != 0:
                        # suppose self.blank == self.num_classes - 1
                        labels = torch.where(labels == 0, self.blank, labels - 1)
                    shortest_paths.append(labels)
            else:
                shortest_paths = []
                # direct iterating does not work as expected
                for i in range(shortest_paths_fsa.shape[0]):
                    aux_labels = shortest_paths_fsa[i].aux_labels.values() if isinstance(shortest_paths_fsa.aux_labels, k2.RaggedInt) else shortest_paths_fsa[i].aux_labels
                    aux_labels = aux_labels[aux_labels != 0][:-1]
                    if self.blank != 0:
                        aux_labels -= 1
                    shortest_paths.append(aux_labels)
            return shortest_paths, scores


class TokenLMDecoder(BaseDecoder):
    """TBD
    """
    def __init__(self,
                 num_classes: int,
                 blank: int,
                 token_lm: Optional[Union[k2.Fsa, str]] = None,
                 intersect_pruned: bool = False,
                 topo_type: str = "default",
                 topo_with_selfloops: bool = True,
                 device: torch.device = torch.device("cpu"),
                 **kwargs
    ):
        super().__init__(num_classes, blank, intersect_pruned, topo_type, topo_with_selfloops, device, **kwargs)
        if token_lm is not None:
            self.token_lm = load_graph(token_lm) if isinstance(token_lm, str) else token_lm
            if self.token_lm is not None:
                self.update_graph(self.token_lm)
            else:
                logging.warning(f"""token_lm was set to None. Use this for debug 
                                purposes only or call .update_graph(token_lm) before using.""")
        else:
            logging.warning(f"""token_lm was set to None. Use this for debug
                            purposes only or call .update_graph(token_lm) before using.""")
            self.token_lm = None
        if self.token_lm is None:
            self.decode_graph = k2.create_fsa_vec([self.ctc_topo_inv.invert()]).to(device)

    def update_graph(self, graph: k2.Fsa):
        self.token_lm = graph
        token_lm = self.token_lm.clone()
        if hasattr(token_lm, 'aux_labels'):
            delattr(token_lm, 'aux_labels')
        if self.pad_fsavec:
            shift_labels_inpl([token_lm], 1)
        labels = token_lm.labels if isinstance(token_lm.labels, torch.Tensor) else token_lm.labels.values()
        if labels.max() != self.ctc_topo_inv.labels.max():
            raise ValueError(f"token_lm is not compatible with the topo: {labels.unique()}, {self.ctc_topo_inv.labels.unique()}")
        self.decode_graph = k2.create_fsa_vec([k2.arc_sort(intersect_with_self_loops(self.ctc_topo_inv, token_lm).invert_())]).to(self.device)
