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

import k2
import torch

import nemo.collections.asr.parts.k2.graph_compilers as graph_compilers
from nemo.collections.asr.parts.k2.topologies import build_topo
from nemo.collections.asr.parts.k2.utils import (
    create_supervision,
    get_arc_weights,
    intersect_with_self_loops,
    invert_permutation,
    load_graph,
    make_blank_first,
    prep_padded_densefsavec,
    shift_labels_inpl,
)
from nemo.utils import logging


class DecoderConf:
    """Graph decoder config.
    Typically contains the pruned intersection parameters.
    """

    def __init__(
        self, search_beam=20.0, output_beam=10.0, min_active_states=30, max_active_states=10000, **kwargs,
    ):
        self.search_beam = search_beam
        self.output_beam = output_beam
        self.min_active_states = min_active_states
        self.max_active_states = max_active_states


class BaseDecoder(object):
    """Base graph decoder with topology for decoding graph.
    Typically uses the same parameters as for the corresponding loss function.
    """

    def __init__(
        self,
        num_classes: int,
        blank: int,
        intersect_pruned: bool = False,
        topo_type: str = "default",
        topo_with_selfloops: bool = True,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        self.num_classes = num_classes
        self.blank = blank
        self.intersect_pruned = intersect_pruned
        self.device = device
        self.topo_type = topo_type
        self.pad_fsavec = self.topo_type == "ctc_compact"
        self.conf = DecoderConf(**kwargs)
        if not hasattr(self, "graph_compiler") or self.graph_compiler is None:
            self.graph_compiler = graph_compilers.CtcTopologyCompiler(
                self.num_classes, self.topo_type, topo_with_selfloops, device
            )
        if not hasattr(self, "base_graph") or self.base_graph is None:
            self.base_graph = k2.create_fsa_vec([self.graph_compiler.ctc_topo_inv.invert()]).to(device)
        self.decoding_graph = None

    def to(self, device: torch.device):
        if self.decoding_graph is not None:
            self.decoding_graph = self.decoding_graph.to(device)
        self.device = device

    def update_graph(self, graph: k2.Fsa):
        raise NotImplementedError

    def decode(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        return_lattices: bool = False,
        return_ilabels: bool = False,
        output_aligned: bool = True,
    ) -> Union[k2.Fsa, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        if self.decoding_graph is None:
            self.decoding_graph = self.base_graph

        if self.blank != 0:
            # rearrange log_probs to put blank at the first place
            # and shift targets to emulate blank = 0
            log_probs, _ = make_blank_first(self.blank, log_probs, None)
        supervisions, order = create_supervision(log_probs_length)
        if self.decoding_graph.shape[0] > 1:
            self.decoding_graph = k2.index_fsa(self.decoding_graph, order).to(device=log_probs.device)

        if log_probs.device != self.device:
            self.to(log_probs.device)
        dense_fsa_vec = (
            prep_padded_densefsavec(log_probs, supervisions)
            if self.pad_fsavec
            else k2.DenseFsaVec(log_probs, supervisions)
        )

        if self.intersect_pruned:
            lats = k2.intersect_dense_pruned(
                a_fsas=self.decoding_graph,
                b_fsas=dense_fsa_vec,
                search_beam=self.conf.search_beam,
                output_beam=self.conf.output_beam,
                min_active_states=self.conf.min_active_states,
                max_active_states=self.conf.max_active_states,
            )
        else:
            indices = torch.zeros(dense_fsa_vec.dim0(), dtype=torch.int32, device=self.device)
            dec_graphs = (
                k2.index_fsa(self.decoding_graph, indices)
                if self.decoding_graph.shape[0] == 1
                else self.decoding_graph
            )
            lats = k2.intersect_dense(dec_graphs, dense_fsa_vec, self.conf.output_beam)
        if self.pad_fsavec:
            shift_labels_inpl([lats], -1)
        self.decoding_graph = None

        if return_lattices:
            lats = k2.index_fsa(lats, invert_permutation(order).to(device=log_probs.device))
            if self.blank != 0:
                # change only ilabels
                # suppose self.blank == self.num_classes - 1
                lats.labels = torch.where(lats.labels == 0, self.blank, lats.labels - 1)
            return lats
        else:
            shortest_path_fsas = k2.index_fsa(
                k2.shortest_path(lats, True), invert_permutation(order).to(device=log_probs.device),
            )
            shortest_paths = []
            probs = []
            # direct iterating does not work as expected
            for i in range(shortest_path_fsas.shape[0]):
                shortest_path_fsa = shortest_path_fsas[i]
                labels = (
                    shortest_path_fsa.labels[:-1].to(dtype=torch.long)
                    if return_ilabels
                    else shortest_path_fsa.aux_labels[:-1].to(dtype=torch.long)
                )
                if self.blank != 0:
                    # suppose self.blank == self.num_classes - 1
                    labels = torch.where(labels == 0, self.blank, labels - 1)
                if not return_ilabels and not output_aligned:
                    labels = labels[labels != self.blank]
                shortest_paths.append(labels[::2] if self.pad_fsavec else labels)
                probs.append(get_arc_weights(shortest_path_fsa)[:-1].to(device=log_probs.device).exp())
            return shortest_paths, probs

    def align(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        return_lattices: bool = False,
        return_ilabels: bool = False,
        output_aligned: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.blank != 0:
            targets = targets + 1
        if self.pad_fsavec:
            targets = targets + 1
        self.decoding_graph = self.graph_compiler.compile(targets, target_lengths)
        return self.decode(
            log_probs,
            log_probs_length,
            return_lattices=return_lattices,
            return_ilabels=return_ilabels,
            output_aligned=output_aligned,
        )


class TokenLMDecoder(BaseDecoder):
    """Graph decoder with token_lm-based decoding graph.
    """

    def __init__(
        self,
        num_classes: int,
        blank: int,
        token_lm: Optional[Union[k2.Fsa, str]] = None,
        intersect_pruned: bool = False,
        topo_type: str = "default",
        topo_with_selfloops: bool = True,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        super().__init__(
            num_classes, blank, intersect_pruned, topo_type, topo_with_selfloops, device, **kwargs,
        )
        if token_lm is not None:
            self.token_lm = load_graph(token_lm) if isinstance(token_lm, str) else token_lm
            if self.token_lm is not None:
                self.update_graph(self.token_lm)
            else:
                logging.warning(
                    f"""token_lm was set to None. Use this for debug 
                                purposes only or call .update_graph(token_lm) before using."""
                )
        else:
            logging.warning(
                f"""token_lm was set to None. Use this for debug
                            purposes only or call .update_graph(token_lm) before using."""
            )
            self.token_lm = None

    def update_graph(self, graph: k2.Fsa):
        self.token_lm = graph
        token_lm = self.token_lm.clone()
        if hasattr(token_lm, "aux_labels"):
            delattr(token_lm, "aux_labels")
        if self.pad_fsavec:
            shift_labels_inpl([token_lm], 1)
        labels = token_lm.labels if isinstance(token_lm.labels, torch.Tensor) else token_lm.labels.values()
        if labels.max() != self.num_classes:
            raise ValueError(f"token_lm is not compatible with the num_classes: {labels.unique()}, {self.num_classes}")
        self.graph_compiler = graph_compilers.CtcNumGraphCompiler(
            self.num_classes, self.topo_type, topo_with_selfloops, device, token_lm
        )
        self.base_graph = k2.create_fsa_vec([self.graph_compiler.base_graph]).to(device)
