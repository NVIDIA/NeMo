# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import NamedTuple, Optional, Union, cast

import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from omegaconf import MISSING, DictConfig, OmegaConf
from tqdm.auto import tqdm
from collections import deque

from nemo.utils import logging


from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.context_biasing.context_graph_universal import ContextGraph, ContextState


class TBranch(NamedTuple):
    """Structure (tuple) to represent a branch in the boosting tree"""

    symbol: int # token id
    start_node: ContextState # start node of the branch
    next_node: ContextState # next node of the branch


@dataclass
class BoostingTreeStorage:
    """
    NumPy-based storage for suffix tree (weighted acceptor) for phrase boosting
    """

    num_states_max: InitVar[int]
    num_arcs_max: InitVar[int]

    vocab_size: int
    max_order: int

    arcs: np.ndarray = field(init=False)
    states: np.ndarray = field(init=False)

    _node_cache: dict[int, int] = field(default_factory=dict)

    unk_score: float = 0.0
    final_eos_score: float = 0.0
    num_states: int = 0
    num_arcs: int = 0
    start_state: int = 0
    bos_state: int = 1

    def __post_init__(self, num_states_max: int, num_arcs_max: int, separate_bos_state: bool = True):
        if max(num_states_max, num_arcs_max) < np.iinfo(np.int32).max:
            int_np_dtype = np.int32
        else:
            int_np_dtype = np.int64
        self.arcs = np.zeros(
            [num_arcs_max],
            dtype=[
                ("from", int_np_dtype),
                ("to", int_np_dtype),
                ("ilabel", int_np_dtype),
                ("weight", np.float32)
            ],
        )
        self.states = np.zeros(
            [num_states_max],
            dtype=[
                ("arcs_start", int_np_dtype),
                ("arcs_end", int_np_dtype),
                ("order", int_np_dtype),
                ("backoff_to", int_np_dtype),
                ("backoff_w", np.float32),
                ("final", np.float32),
            ],
        )
        self.states["final"] = self.final_eos_score
        self.bos_state = 1 if separate_bos_state else self.start_state
        self._node_cache[0] = 0
        self.separate_bos_state = False

    def _add_tbranches_first_order(self, tbranches: list):
        """Add all first order tbranches to the model (similar with unigrams for N-Gram LM)"""

        tbranches = sorted(tbranches, key=lambda x: (x.start_node.id, x.symbol))

        self.num_states = 1
        self.num_arcs = 0
        # state: start_arcs, end_arcs, order, backoff_to, backoff_weight
        self.states[self.start_state] = (0, self.vocab_size, 1, self.start_state, 0.0, 0.0)
        added_symbols = set()
        num_vocab_labels = 0
        for tbranch in tbranches:
            ilabel = tbranch.symbol
            assert ilabel < self.vocab_size
            arc_id = ilabel
            added_symbols.add(ilabel)
            next_state = self.num_states
            self.num_states += 1
            self.arcs[arc_id] = (self.start_state, next_state, ilabel, tbranch.next_node.token_score)
            self.num_arcs += 1

            # TODO: do we need to increase arc weigth in the case of the final node (end of phrase)?
            if tbranch.next_node.is_end:
                backoff_weight = 0.0
            else:
                backoff_weight = tbranch.next_node.fail.node_score - tbranch.next_node.node_score

            # state order
            self.states[next_state] = (
                0,
                0,
                self.states[self.start_state]["order"] + 1,
                self.start_state,
                backoff_weight,
                self.final_eos_score if tbranch.next_node.is_end else 0.0,
            )
            num_vocab_labels += 1
            self._node_cache[tbranch.next_node.id] = next_state

        for ilabel in range(self.vocab_size):
            if ilabel not in added_symbols:
                self.arcs[ilabel] = (self.start_state, self.start_state, ilabel, self.unk_score)
                self.num_arcs += 1


    def _add_tbranches_next_order(self, tbranches: list):
        """Add tbranches for the order > 1; should be called after adding first order tokens (unigrams), using increasing order"""
        tbranches = sorted(tbranches, key=lambda x: (x.start_node.id, x.symbol))

        for tbranch in tqdm(tbranches):
            ilabel = tbranch.symbol
            from_state = self._node_cache[tbranch.start_node.id]
            assert ilabel < self.vocab_size
            backoff_state = self._node_cache[tbranch.next_node.fail.id]

            # TODO: do we need to increase arc weigth in the case of the final node (end of phrase)?
            if tbranch.next_node.is_end and not self.uniform_weights:
                backoff_weight = tbranch.next_node.fail.node_score
            else:
                backoff_weight = tbranch.next_node.fail.node_score - tbranch.next_node.node_score
            
            arc_id = self.num_arcs
            next_state = self.num_states
            self.num_arcs += 1
            self.num_states += 1
            token_score = tbranch.next_node.token_score
            if self.uniform_weights and tbranch.next_node.is_end:
                token_score += tbranch.next_node.node_score

            self.arcs[arc_id] = (from_state, next_state, ilabel, token_score)

            self.states[next_state] = (
                0,
                0,
                self.states[from_state]["order"] + 1,
                backoff_state,
                backoff_weight,
                self.final_eos_score if tbranch.next_node.is_end else 0.0,
            )
            
            self._node_cache[tbranch.next_node.id] = next_state
            
            if self.states[from_state]["arcs_start"] == 0:
                self.states[from_state]["arcs_start"] = arc_id
                self.states[from_state]["arcs_end"] = arc_id + 1
            else:
                assert self.states[from_state]["arcs_end"] == arc_id
                self.states[from_state]["arcs_end"] = arc_id + 1


    def _start_adding_tbranches_for_order(self, order: int):
        """Prepare for adding tbranches for the given order: initialize temporary storage"""
        self._start_arcs = self.num_arcs
        self._cur_order = order
        self._tbranches = []
        self._tbranches_cnt = 0


    def _end_adding_tbranches_for_order(self, order: int):
        """Finish adding tbranches for the given order"""
        if order == 1:
            assert len(self._tbranches) == self._tbranches_cnt
            self._add_tbranches_first_order(tbranches=self._tbranches)
            self._tbranches = None
            self._tbranches_cnt = 0
        else:
            assert len(self._tbranches) == self._tbranches_cnt
            self._add_tbranches_next_order(tbranches=self._tbranches)
            self._tbranches = None
            self._tbranches_cnt = 0


    def sanity_check(self):
        """Sanity check for the model"""
        assert (self.arcs["ilabel"][: self.num_arcs] < self.vocab_size).all()
        assert (self.arcs["ilabel"][: self.num_arcs] >= 0).all()


@dataclass
class BoostingTreeConfig:
    """
    N-Gram LM Config
    """

    num_states: int = MISSING
    num_arcs: int = MISSING
    max_order: int = MISSING
    vocab_size: int = MISSING
    separate_bos_state: bool = False
    use_triton: bool | None = None


class GPUBoostingTreeModel(NGramGPULanguageModel):
    """
    GPU-accelerated boosting tree supporting batched queries.
    Fast implementation for parallel queries for full vocabulary.
    Supports autograd (differentiable weights).
    """

    START_STATE = 0

    def __init__(
        self,
        cfg: DictConfig,
        trainer: Trainer = None,
    ):
        """
        Stubs for constructor that does not initialize the structure.
        This constructor can be useful when storing/loading module using native torch serialization mechanism
        instead of directly reading ARPA model -> converting to Torch, which can be slow for large N-Gram models
        (of several GBs).

        Args:
            cfg:
                num_states: number of states in graph
                num_arcs: number of arcs (transitions) in graph
                max_order: maximum order of n-gram LM (maximum possible nubmer of transitions without backoffs)
                vocab_size: vocabulary size (existing vocabulary units in LM; should not include blank etc.)
                separate_bos_state: separate Begin-of-Sentence state (default: True - for n-gram LM)
                use_triton: allow using Triton implementation;
                    None (default) means "auto" (used if available), True means forced mode
                    (will crash if Triton is unavailable)
            trainer: Lightning trainer (optional)
        """
        super().__init__(cfg=cfg, trainer=trainer)
        self.bos_state = self.START_STATE # Always START_STATE for gpu boosting tree

    
    @classmethod
    def _read_cb_tree(
        cls,
        cb_tree: ContextGraph,
    ) -> tuple[dict[int, int], list[TBranch]]:
        """
        Read context-biasing tree from python structure and return branches in TBranch format.

        Args:
            cb_tree: python context-biasing tree
        """

        seen = set()
        queue = deque()
        queue.append(cb_tree.root)
        seen.add(0)
        order2cnt = {}
        tbranches_list = []

        # read context graph tree in breadth-first order to add branches for boosting tree generation
        while len(queue):
            current_node = queue.popleft()
            for token, node in current_node.next.items():
                if node.id not in seen:
                    tbranches_list.append(TBranch(symbol=token, start_node=current_node, next_node=node))
                    order2cnt[node.level] = order2cnt.get(node.level, 0) + 1
                    queue.append(node)
        
        return order2cnt, tbranches_list

    
    @classmethod
    def from_cb_tree(
        cls,
        cb_tree: ContextGraph,
        vocab_size: int,
        unk_score: float = True,
        final_eos_score: float = 0.0,
        use_triton: bool | None = None,
        uniform_weights: bool | None = None,
    ) -> "GPUBoostingTreeModel":
        """
        Constructor from Icefall context graph (dict-based tree).

        Args:
            cb_tree: context-biasing graph
            vocab_size: vocabulary size (existing vocabulary units in LM; should not include blank etc.)
            unk_score: score for unknown tokens
            final_eos_score: score for eos token after detected end of context phrase
            use_triton: allow using Triton implementation;
                None (default) means "auto" (used if available), True means forced mode
                (will crash if Triton is unavailable)

        Returns:
            GPUBoostingTreeModel instance
        """
        logging.info(f"{cls.__name__}: reading boosting tree from {cb_tree}")

        order2cnt, tbranches_list = cls._read_cb_tree(cb_tree=cb_tree)

        # init suffix tree storage
        max_states = cb_tree.num_nodes + 1 # + 1 for root state
        boosting_tree_np = BoostingTreeStorage(
            num_states_max=max_states,
            num_states=0,
            num_arcs=0,
            num_arcs_max=max_states * 2 + vocab_size * 2 + 1,
            unk_score=unk_score,
            final_eos_score=final_eos_score,
            vocab_size=vocab_size,
            max_order=max(order2cnt)+1,
        )

        boosting_tree_np.uniform_weights = uniform_weights
        # convert cb_tree to np boosting tree
        tbranch_cur_order_i = 0
        cur_order = 1

        for tbranch in tqdm(tbranches_list, total=len(tbranches_list)):

            if tbranch_cur_order_i == 0:
                boosting_tree_np._start_adding_tbranches_for_order(order=cur_order)
            tbranch_cur_order_i += 1

            # add tbranch
            boosting_tree_np._tbranches.append(tbranch)
            boosting_tree_np._tbranches_cnt += 1
            
            if tbranch_cur_order_i == order2cnt[cur_order]:
                boosting_tree_np._end_adding_tbranches_for_order(order=cur_order)
                logging.info(f"Processed {order2cnt[cur_order]} n-grams of order {cur_order}")
                cur_order += 1
                tbranch_cur_order_i = 0

        assert tbranch_cur_order_i == 0
        boosting_tree_np.sanity_check()

        return GPUBoostingTreeModel.from_boosting_tree_np(boosting_tree_np=boosting_tree_np, use_triton=use_triton)


    @classmethod
    def from_boosting_tree_np(
        cls, boosting_tree_np: BoostingTreeStorage, use_triton: bool | None = None
    ) -> "GPUBoostingTreeModel":
        """
        Constructor from suffix tree storage.

        Args:
            suffix_tree_np: suffix tree
            use_triton: allow using Triton implementation;
                None (default) means "auto" (used if available), True means forced mode
                (will crash if Triton is unavailable)

        Returns:
            GPUBoostingTreeModel instance
        """
        model = GPUBoostingTreeModel(
            OmegaConf.structured(
                BoostingTreeConfig(
                    num_states=boosting_tree_np.num_states,
                    num_arcs=boosting_tree_np.num_arcs,
                    max_order=boosting_tree_np.max_order,
                    vocab_size=boosting_tree_np.vocab_size,
                    use_triton=use_triton,
                )
            )
        )
        model._init_from_suffix_tree_np(suffix_tree_np=boosting_tree_np)
        model._resolve_final()
        return model


    def advance(self, states: torch.Tensor, eos_id: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance `states` [B]: return scores [B, V] and next states [B, V] for full vocab
        Args:
            states: batch of states
            eos_id: if not None, for eos symbol use final state weight

        Returns:
            tuple with next states and scores
        """
        if self.use_triton and states.device.type == "cuda":
            # raise NotImplementedError("Triton implementation is not available yet")
            scores, next_states = self._advance_triton(states=states)
        else:
            # raise NotImplementedError("Pytorch implementation is not available yet")
            scores, next_states = self._advance_pytorch(states=states)
        
        # replace eos_id score with maximum state weight to prevent the model from hallucinating
        if eos_id is not None:
            # 1. replace eos score with maximum boosting value at each step
            scores[:, eos_id] = torch.clamp(torch.max(scores, dim=1).values, min=0.0)
            
            # 2. increase eos score after detected end of context phrase
            scores[:, eos_id] += self.get_final(states)
            
            next_states[:, eos_id] = states
        return scores, next_states


    def get_final(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get final weights for states

        Args:
            states: batch of states

        Returns:
            tensor [B] with final weights for each state
        """

        return self.final_weights[states]