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

from nemo.collections.asr.parts.k2.utils import build_ctc_topo
from nemo.collections.asr.parts.k2.utils import create_supervision
from nemo.collections.asr.parts.k2.utils import intersect_with_self_loops
from nemo.collections.asr.parts.k2.utils import invert_permutation
from nemo.collections.asr.parts.k2.utils import make_blank_first
from nemo.collections.asr.parts.k2.utils import load_graph
from nemo.utils import logging


class DecoderConf:
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
    def __init__(self,
                 num_classes: int,
                 blank: int,
                 pruned: bool = False,
                 topo_type: str = "full",
                 device: torch.device = torch.device("cpu"),
                 **kwargs
    ):
        self.num_classes = num_classes
        self.blank = blank
        self.pruned = pruned
        self.device = device
        if topo_type != "full":
            raise NotImplementedError("Not implemented yet")
        else:
            self.ctc_topo_inv = k2.arc_sort(build_ctc_topo(list(range(num_classes))).invert_())
        self.decode_graph = None
        self.conf = DecoderConf(**kwargs)

    def to(self, device: torch.device):
        if self.decode_graph is not None:
            self.decode_graph = self.decode_graph.to(device)
        self.device = device

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
        dense_fsa_vec = k2.DenseFsaVec(log_probs, supervisions)

        if self.pruned:
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

        if return_lattices:
            lats = k2.index_fsa(lattice_orig_med, invert_permutation(order).to(dtype=torch.int32, device=input_lengths.device))
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
                shortest_paths = torch.tensor([path.labels[:-1] for path in shortest_paths_fsa], device=shortest_paths_fsa.device)
                if self.blank != 0:
                    # suppose self.blank == self.num_classes - 1
                    shortest_paths = torch.where(shortest_paths == 0, self.blank, shortest_paths - 1)
                shortest_paths = [path for path in shortest_paths]
            else:
                shortest_paths = []
                # direct iterating is bugged
                for i in range(shortest_paths_fsa.shape[0]):
                    aux_labels = shortest_paths_fsa[i].aux_labels.values() if isinstance(shortest_paths_fsa.aux_labels, k2.RaggedInt) else shortest_paths_fsa[i].aux_labels
                    aux_labels = aux_labels[aux_labels != 0][:-1]
                    if self.blank != 0:
                        aux_labels -= 1
                    shortest_paths.append(aux_labels)
            return shortest_paths, scores


class TokenLMDecoder(BaseDecoder):
    def __init__(self,
                 num_classes: int,
                 blank: int,
                 token_lm: Optional[Union[k2.Fsa, str]],
                 pruned: bool = False,
                 topo_type: str = "full",
                 device: torch.device = torch.device("cpu"),
                 **kwargs
    ):
        super().__init__(num_classes, blank, pruned, topo_type, device, **kwargs)
        if token_lm is None:
            logging.warning(f"""token_lm was set to None. 
                            Use this for debug purposes only.""")
            self.token_lm = token_lm
            self.decode_graph = k2.create_fsa_vec([self.ctc_topo_inv.invert_()]).to(device)
        else:
            self.token_lm = load_graph(token_lm) if isinstance(token_lm, str) else token_lm
            self.decode_graph = k2.create_fsa_vec([intersect_with_self_loops(self.ctc_topo_inv, self.token_lm).invert_()]).to(device)


class TLGDecoder(BaseDecoder):
    def __init__(self,
                 num_classes: int,
                 blank: int,
                 L: Union[k2.Fsa, str],
                 G: Union[k2.Fsa, str],
                 pruned: bool = False,
                 topo_type: str = "full",
                 device: torch.device = torch.device("cpu"),
                 **kwargs
    ):
        super().__init__(num_classes, blank, pruned, topo_type, device, **kwargs)
        self.L = load_graph(token_lm) if isinstance(L, str) else L
        self.G = load_graph(token_lm) if isinstance(G, str) else G
        raise NotImplementedError
        self.decode_graph = k2.create_fsa_vec([]).to(device)
