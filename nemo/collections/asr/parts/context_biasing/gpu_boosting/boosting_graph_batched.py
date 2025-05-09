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
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from collections import deque

from nemo.collections.asr.parts.submodules.ngram_lm.constants import DEFAULT_TOKEN_OFFSET
from nemo.collections.common.parts import NEG_INF
from nemo.core import ModelPT, PretrainedModelInfo
from nemo.core.utils.optional_libs import TRITON_AVAILABLE, triton_required
from nemo.utils import logging

if TRITON_AVAILABLE:
    import triton
    from nemo.collections.asr.parts.submodules.ngram_lm.ngram_lm_triton import ngram_advance_triton_kernel

from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.context_biasing.gpu_boosting.context_graph import ContextGraph, ContextState


class TBranch(NamedTuple):
    """Structure (tuple) to represent N-Gram element (symbol, weight, backoff)"""

    symbol: int
    start_node: ContextState
    next_node: ContextState


class Arc(NamedTuple):
    """Structure (tuple) to represent arc in the weighted acceptor"""

    weight: float
    ilabel: int
    to: int


@dataclass
class SuffixTreeStorage:
    """
    NumPy-based storage for suffix tree (weighted acceptor) for N-Gram LM
    """

    num_states_max: InitVar[int]
    num_arcs_max: InitVar[int]

    vocab_size: int
    max_order: int

    separate_bos_state: InitVar[bool] = True

    arcs: np.ndarray = field(init=False)
    states: np.ndarray = field(init=False)

    _node_cache: dict[int, int] = field(default_factory=dict)

    unk_score: float = 0.0
    eos_id: Optional[int] = None

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
        self.states["final"] = NEG_INF
        self.bos_state = 1 if separate_bos_state else self.start_state
        self._node_cache[0] = 0

    def _add_tbranches_first_order(self, tbranches: list):
        """Add all first order tbranches to the model (similar with unigrams for N-Gram LM)"""

        tbranches = sorted(tbranches, key=lambda x: (x.start_node.id, x.symbol))

        self.num_states = 1
        self.num_arcs = 0
        # state: start_arcs, end_arcs, order, backoff_to, backoff_weight
        self.states[self.start_state] = (0, self.vocab_size, 1, self.start_state, 0.0, NEG_INF)
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
                NEG_INF,
            )
            num_vocab_labels += 1
            self._node_cache[tbranch.next_node.id] = next_state

        for ilabel in range(self.vocab_size):
            if ilabel not in added_symbols:
                if self.eos_id is not None and ilabel == self.eos_id:
                    # TODO: add separate score for EOS token
                    self.arcs[ilabel] = (self.start_state, self.start_state, ilabel, self.unk_score)
                else:
                    self.arcs[ilabel] = (self.start_state, self.start_state, ilabel, self.unk_score)
                self.num_arcs += 1


    def _add_tbranches_next_order(self, tbranches: list):
        """Add tbranches for the order > 1; should be called after adding unigrams, using increasing order"""
        tbranches = sorted(tbranches, key=lambda x: (x.start_node.id, x.symbol))

        for tbranch in tqdm(tbranches):
            ilabel = tbranch.symbol
            from_state = self._node_cache[tbranch.start_node.id]
            assert ilabel < self.vocab_size
            backoff_state = self._node_cache[tbranch.next_node.fail.id]

            # TODO: do we need to increase arc weigth in the case of the final node (end of phrase)?
            if tbranch.next_node.is_end:
                backoff_weight = 0.0
            else:
                backoff_weight = tbranch.next_node.fail.node_score - tbranch.next_node.node_score

            arc_id = self.num_arcs
            next_state = self.num_states
            self.num_arcs += 1
            self.num_states += 1
            self.arcs[arc_id] = (from_state, next_state, ilabel, tbranch.next_node.token_score)

            self.states[next_state] = (
                0,
                0,
                self.states[from_state]["order"] + 1,
                backoff_state,
                backoff_weight,
                NEG_INF,
            )
            
            self._node_cache[tbranch.next_node.id] = next_state
            
            if self.states[from_state]["arcs_start"] == 0:
                self.states[from_state]["arcs_start"] = arc_id
                self.states[from_state]["arcs_end"] = arc_id + 1
            else:
                assert self.states[from_state]["arcs_end"] == arc_id
                self.states[from_state]["arcs_end"] = arc_id + 1


    def _start_adding_tbranches_for_order(self, order: int, max_tbranches: int):
        """Prepare for adding tbranches for the given order: initialize temporary storage"""
        self._start_arcs = self.num_arcs
        self._cur_order = order
        self._tbranches = []
        self._tbranches_cnt = 0


    def _end_adding_tbranches_for_order(self, order: int):
        """Finish adding ngrams for the given order"""
        if order == 1:
            assert len(self._tbranches) == self._tbranches_cnt
            self._add_tbranches_first_order(tbranches=self._tbranches)
            self._tbranches = None
            self._tbranches_cnt = 0
        # elif order < self.max_order:
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
        cfg = cast(BoostingTreeConfig, cfg)
        self.use_triton = cfg.use_triton if cfg.use_triton is not None else TRITON_AVAILABLE
        if not self.use_triton:
            logging.warning(
                "Triton is disabled. Version without Triton is not compatible with Cuda graphs; decoding can be slow"
            )

        self.bos_state = 1 if cfg.separate_bos_state else self.START_STATE
        self.vocab_size = cfg.vocab_size
        self.num_states = cfg.num_states
        self.num_arcs = cfg.num_arcs
        self.max_order = cfg.max_order
        self.num_arcs_extended = cfg.num_arcs + self.vocab_size  # + extra padding

        # parameters: weights (forward/backoff/final)
        self.arcs_weights = nn.Parameter(torch.zeros([self.num_arcs_extended]))
        self.backoff_weights = nn.Parameter(torch.zeros([self.num_states]))
        self.final_weights = nn.Parameter(torch.zeros([self.num_states]))

        if max(self.num_states, self.num_arcs_extended) < torch.iinfo(torch.int32).max:
            int_dtype = torch.int32
        else:
            int_dtype = torch.int64
        # buffers: LM (suffix tree) structure
        # arcs data
        self.register_buffer("from_states", torch.zeros([self.num_arcs_extended], dtype=int_dtype))
        self.register_buffer("to_states", torch.zeros([self.num_arcs_extended], dtype=int_dtype))
        self.register_buffer("ilabels", torch.zeros([self.num_arcs_extended], dtype=int_dtype))

        # states data
        self.register_buffer("backoff_to_states", torch.zeros([self.num_states], dtype=int_dtype))
        self.register_buffer("start_end_arcs", torch.zeros([self.num_states, 2], dtype=int_dtype))
        self.register_buffer("state_order", torch.zeros([self.num_states], dtype=int_dtype))

        self._final_resolved = False

    @classmethod
    def list_available_models(cls) -> list[PretrainedModelInfo]:
        """Stub necessary to create the ModelPT. Not used for LM"""
        return []

    def setup_training_data(self, train_data_config: Union[DictConfig, dict]):
        """Stub necessary to create the ModelPT. Not used for LM"""
        pass

    def setup_validation_data(self, val_data_config: Union[DictConfig, dict]):
        """Stub necessary to create the ModelPT. Not used for LM"""
        pass

    @classmethod
    def from_nemo(
        cls,
        lm_path: Path | str,
        vocab_size: int,
        use_triton: bool | None = None,
    ) -> "GPUBoostingTreeModel":
        """
        Constructor from Nemo checkpoint (state dict).

        Args:
            lm_path: path to .nemo checkpoint
            vocab_size: model vocabulary size
            use_triton: allow using Triton implementation; None (default) means "auto" (used if available)
        """
        model = GPUBoostingTreeModel.restore_from(restore_path=str(lm_path), map_location="cpu")
        model._resolve_final()
        assert model.vocab_size == vocab_size
        model.use_triton = use_triton if use_triton is not None else TRITON_AVAILABLE
        if not model.use_triton:
            logging.warning(
                "Triton is disabled. Version without Triton is not compatible with Cuda graphs; decoding can be slow"
            )
        return model

    
    @classmethod
    def _read_cb_tree(
        cls,
        cb_tree: ContextGraph,
    ) -> tuple[dict[int, int], list[TBranch]]:
        """
        Read context-biasing graph from Icefall and return branches in TBranch format.

        Args:
            cb_tree: context graph
        """

        seen = set()
        queue = deque()
        queue.append(cb_tree.root)
        seen.add(0)
        order2cnt = {}
        tbranches_list = []

        # read context graph tree in breadth-first order to add branches for suffix tree generation
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
        eos_id: Optional[int] = None,
        use_triton: bool | None = None,
    ) -> "GPUBoostingTreeModel":
        """
        Constructor from Icefall context graph (dict-based tree).

        Args:
            cb_tree: context-biasing graph
            vocab_size: vocabulary size (existing vocabulary units in LM; should not include blank etc.)
            unk_score: score for unknown tokens
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
        suffix_tree_np = SuffixTreeStorage(
            num_states_max=max_states,
            num_states=0,
            num_arcs=0,
            num_arcs_max=max_states * 2 + vocab_size * 2 + 1,
            unk_score=unk_score,
            eos_id=eos_id,
            vocab_size=vocab_size,
            max_order=max(order2cnt)+1,
        )

        # convert cb_tree to np suffix tree
        tbranch_cur_order_i = 0
        cur_order = 1

        for tbranch in tqdm(tbranches_list, total=len(tbranches_list)):

            if tbranch_cur_order_i == 0:
                suffix_tree_np._start_adding_tbranches_for_order(order=cur_order, max_tbranches=order2cnt[cur_order])
            tbranch_cur_order_i += 1

            # add tbranch
            suffix_tree_np._tbranches.append(tbranch)
            suffix_tree_np._tbranches_cnt += 1
            
            if tbranch_cur_order_i == order2cnt[cur_order]:
                suffix_tree_np._end_adding_tbranches_for_order(order=cur_order)
                logging.info(f"Processed {order2cnt[cur_order]} n-grams of order {cur_order}")
                cur_order += 1
                tbranch_cur_order_i = 0

        assert tbranch_cur_order_i == 0
        suffix_tree_np.sanity_check()

        return GPUBoostingTreeModel.from_suffix_tree(suffix_tree_np=suffix_tree_np, use_triton=use_triton)


    @classmethod
    def from_suffix_tree(
        cls, suffix_tree_np: SuffixTreeStorage, use_triton: bool | None = None
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
                    num_states=suffix_tree_np.num_states,
                    num_arcs=suffix_tree_np.num_arcs,
                    max_order=suffix_tree_np.max_order,
                    vocab_size=suffix_tree_np.vocab_size,
                    use_triton=use_triton,
                )
            )
        )
        model._init_from_suffix_tree_np(suffix_tree_np=suffix_tree_np)
        model._resolve_final()
        return model


    def _init_from_suffix_tree_np(self, suffix_tree_np: SuffixTreeStorage):
        """Helper function to init params from suffix tree params"""
        # parameters: weights
        self.arcs_weights.data.copy_(torch.from_numpy(suffix_tree_np.arcs["weight"][: self.num_arcs_extended]))
        self.backoff_weights.data.copy_(torch.from_numpy(suffix_tree_np.states["backoff_w"][: self.num_states]))
        self.final_weights.data.copy_(torch.from_numpy(suffix_tree_np.states["final"][: self.num_states]))

        # buffers: LM (suffix tree) structure
        self.from_states.data.copy_(torch.from_numpy(suffix_tree_np.arcs["from"][: self.num_arcs_extended]))
        self.to_states.data.copy_(torch.from_numpy(suffix_tree_np.arcs["to"][: self.num_arcs_extended]))
        self.ilabels.data.copy_(torch.from_numpy(suffix_tree_np.arcs["ilabel"][: self.num_arcs_extended]))
        self.backoff_to_states.data.copy_(torch.from_numpy(suffix_tree_np.states["backoff_to"][: self.num_states]))

        self.start_end_arcs.data[:, 0].copy_(torch.from_numpy(suffix_tree_np.states["arcs_start"][: self.num_states]))
        self.start_end_arcs.data[:, 1].copy_(torch.from_numpy(suffix_tree_np.states["arcs_end"][: self.num_states]))
        self.state_order.data.copy_(torch.from_numpy(suffix_tree_np.states["order"][: self.num_states]))

        # sanity check
        # import pdb; pdb.set_trace()
        assert self.state_order.min().item() == 1
        assert self.state_order.max().item() <= self.max_order

    def get_init_states(self, batch_size: int, bos=True) -> torch.Tensor:
        """
        Get batch of the initial states

        Args:
            batch_size: batch size
            bos: use begin-of-sentence state

        Returns:
            tensor [B] of initial states
        """
        device = self.arcs_weights.device
        return torch.full(
            [batch_size], fill_value=self.bos_state if bos else self.START_STATE, device=device, dtype=torch.long
        )

    def forward(
        self,
        labels: torch.Tensor,
        labels_lengths: Optional[torch.Tensor] = None,
        bos: bool = True,
        eos: bool = False,
    ) -> torch.Tensor:
        """
        Compute log-probabilities for all labels in utterances using N-Gram LM.

        Args:
            labels: label sequences [B x L] if eos=False, [B x (L+1)] if eos=True
            labels_lengths (optional): lengths of the label sequences
            bos: start with BOS symbol
            eos: add EOS score after the sentence

        Returns:
            Tensor [B x L] with scores for each label in the utterance
        """
        return self.score_sentences(labels=labels, labels_lengths=labels_lengths, bos=bos, eos=eos)

    def score_sentences(
        self,
        labels: torch.Tensor,
        labels_lengths: Optional[torch.Tensor] = None,
        bos: bool = True,
        eos: bool = False,
    ) -> torch.Tensor:
        """
        Compute log-probabilities for all labels in utterances using N-Gram LM.

        Args:
            labels: label sequences [B x L] if eos=False, [B x (L+1)] if eos=True
            labels_lengths (optional): lengths of the label sequences
            bos: start with BOS symbol
            eos: add EOS score after the sentence

        Returns:
            Tensor [B x (L + 1) if eos else B x L] with scores for each label in the utterance
        """
        device = labels.device
        batch_size, max_length = labels.shape
        if labels_lengths is None:
            labels_lengths = torch.full([batch_size], fill_value=max_length, dtype=torch.int32, device=device)
        batch_size, max_length = labels.shape
        scores = torch.zeros([batch_size, max_length + (1 if eos else 0)], device=device)
        states = self.get_init_states(batch_size=batch_size, bos=bos)
        # NB: It is possible to speedup this algorithm with a custom kernel (no need to retrieve all weights/labels)
        for i in range(max_length):
            # NB: _advance_triton is not differentiable (need to implement backward manually);
            # for training _advance_pytorch only can be used
            prev_states = states
            step_scores, states = self._advance_pytorch(states)
            scores[:, i] = step_scores.gather(dim=1, index=labels[:, i].unsqueeze(-1)).squeeze(-1) * (
                i < labels_lengths
            )
            # get next states, preserve last state if the utterance ended
            states = torch.where(
                i < labels_lengths, states.gather(dim=1, index=labels[:, i].unsqueeze(-1)).squeeze(-1), prev_states
            )
        if eos:
            final_weights = self.get_final(states)
            scores.scatter_(dim=1, index=labels_lengths.unsqueeze(-1).to(torch.int64), src=final_weights.unsqueeze(-1))
        return scores

    def advance(self, states: torch.Tensor, eos_id: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance `states` [B]: return scores [B, V] and next states [B, V] for full vocab
        Args:
            states: batch of states
            eos_id: if not None, for eos symbol use final state weight

        Returns:
            tuple with next states and scores
        """
        self.use_triton = False
        if self.use_triton and states.device.type == "cuda":
            scores, next_states = self._advance_triton(states=states)
        else:
            scores, next_states = self._advance_pytorch(states=states)

        # replace weight corresponding to eos_id with final state weight
        # if eos_id is not None:
        #     scores[:, eos_id] = self.get_final(states=states)
        #     next_states[:, eos_id] = states
        return scores, next_states

    def _advance_pytorch(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance `states` [B]: return scores [B, V] and next states [B, V] for full vocab.
        PyTorch implementation (slow, differentiable).

        Args:
            states: batch of states

        Returns:
            tuple of scores and next states
        """
        batch_size = states.shape[0]
        device = states.device
        current_states = states.clone()
        states_dtype = current_states.dtype

        # init output tensors
        out_scores = torch.zeros(batch_size, self.vocab_size, device=device)
        out_states = torch.full([batch_size, self.vocab_size], fill_value=-1, dtype=states_dtype, device=device)

        # helper ranges
        vocab_range = torch.arange(self.vocab_size, device=device)
        batch_indices = torch.arange(batch_size, device=device)

        # backoff weight accumulator
        accumulated_backoff = torch.zeros(batch_size, device=device)
        # loop condition
        start_state_not_processed = torch.full([batch_size], fill_value=True, dtype=torch.bool, device=device)

        num_iterations = 0
        while start_state_not_processed.any():
            assert num_iterations <= self.max_order, "Infinite loop in LM advance"
            num_iterations += 1
            # get arc boundaries
            # import ipdb; ipdb.set_trace()
            start, end = self.start_end_arcs[current_states].unbind(dim=1)
            # number of arcs for each state cannot be larger than vocab size
            indices = start[:, None] + vocab_range[None, :]
            mask = indices < end[:, None]
            mask &= start_state_not_processed[:, None]
            mask_flat = mask.view(-1)
            indices_flat = indices.view(-1)
            # map indices outside the mask to vocab_size + 1
            scores_add = torch.zeros([batch_size, self.vocab_size + 1], device=device, dtype=out_scores.dtype)
            out_states_add = torch.full(
                [batch_size, self.vocab_size + 1], fill_value=-1, device=device, dtype=states_dtype
            )
            ilabels = self.ilabels[indices_flat] * mask_flat + ~mask_flat * self.vocab_size
            scores_add[batch_indices.repeat_interleave(self.vocab_size), ilabels] = self.arcs_weights[indices_flat]
            out_states_add[batch_indices.repeat_interleave(self.vocab_size), ilabels] = self.to_states[
                indices_flat
            ].to(states_dtype)
            # fill out_scores and out_states with new values where state is not found yet
            state_found = out_states != -1
            out_scores = torch.where(
                state_found, out_scores, accumulated_backoff.unsqueeze(-1) + scores_add[:, : self.vocab_size]
            )
            out_states = torch.where(state_found, out_states, out_states_add[:, : self.vocab_size])
            # update loop condition; process backoffs
            start_state_not_processed &= current_states != self.START_STATE
            accumulated_backoff += self.backoff_weights[current_states] * start_state_not_processed
            torch.where(
                start_state_not_processed, self.backoff_to_states[current_states], current_states, out=current_states
            )
        return out_scores, out_states

    @triton_required
    def _advance_triton(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance `states` [B]: return scores [B, V] and next states [B, V] for full vocab.
        Triton implementation. Currently not differentiable.

        Args:
            states: batch of states

        Returns:
            tuple of scores and next states
        """
        batch_size = states.shape[0]
        device = states.device
        scores = torch.empty([batch_size, self.vocab_size], device=device, dtype=self.arcs_weights.dtype)
        new_states = torch.empty([batch_size, self.vocab_size], dtype=torch.long, device=device)

        ngram_advance_triton_kernel[batch_size,](
            vocab_size=self.vocab_size,
            states_ptr=states,
            new_states_ptr=new_states,
            scores_ptr=scores,
            start_state=self.START_STATE,
            to_states_ptr=self.to_states,
            ilabels_ptr=self.ilabels,
            arcs_weights_ptr=self.arcs_weights,
            start_end_arcs_ptr=self.start_end_arcs,
            backoff_to_states_ptr=self.backoff_to_states,
            backoff_weights_ptr=self.backoff_weights,
            BLOCK_SIZE=triton.next_power_of_2(self.vocab_size),
        )

        return scores, new_states

    def get_final(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get final weights for states

        Args:
            states: batch of states

        Returns:
            tensor [B] with final weights for each state
        """
        if self._final_resolved:
            return self.final_weights[states]
        logging.warning("Final weights are not resolved; using slow implementation")
        return self._get_final_pytorch(states=states)

    def _resolve_final(self):
        """Resolve final weights for all states by iterating over backoffs"""
        if self._final_resolved:
            return
        with torch.no_grad():
            self.final_weights.data.copy_(
                self._get_final_pytorch(states=torch.arange(self.num_states, device=self.final_weights.device))
            )
        self._final_resolved = True

    def _get_final_pytorch(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get final weights for states, resolving backoffs

        Args:
            states: batch of states

        Returns:
            batch of final weights
        """
        cur_states = states.clone().detach()
        out_scores = self.final_weights[cur_states]
        accumulated_backoff = torch.zeros_like(out_scores)
        while (out_scores <= NEG_INF).any() and (cur_states != self.START_STATE).any():
            accumulated_backoff += self.backoff_weights[cur_states]
            cur_states = self.backoff_to_states[cur_states]
            cur_final = self.final_weights[cur_states]
            out_scores = torch.where(
                (out_scores > NEG_INF) | (cur_final <= NEG_INF), out_scores, accumulated_backoff + cur_final
            )
        return out_scores
