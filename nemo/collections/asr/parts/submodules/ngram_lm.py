# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from nemo.core.utils.optional_libs import KENLM_AVAILABLE, TRITON_AVAILABLE, kenlm_required, triton_required
from nemo.utils import logging

if KENLM_AVAILABLE:
    import kenlm

if TRITON_AVAILABLE:
    import triton

    from nemo.collections.asr.parts.submodules.ngram_lm_triton import _ngram_triton_kernel


def _log_10_to_e(score):
    return score / np.log10(np.e)


class KenLMWrapper:
    """
    KenLM model wrapper for single element and batched queries for decoding (reference) and testing purposes.
    """

    @kenlm_required
    def __init__(self, model_path: Path | str, token_offset=100):
        self.ngram_lm = kenlm.Model(str(model_path))
        self.token_offset = token_offset

    def get_init_state(self, bos=True):
        init_lm_state = kenlm.State()

        if not bos:
            return init_lm_state

        self.ngram_lm.BeginSentenceWrite(init_lm_state)
        return init_lm_state

    def compute_scores_batch(
        self, states: list["kenlm.State"], vocab_size: int
    ) -> tuple[torch.Tensor, list[list["kenlm.State"]]]:
        batch_size = len(states)
        new_states = [[] for _ in range(len(states))]
        scores = torch.zeros(batch_size, vocab_size)
        for i, state in enumerate(states):
            for label in range(vocab_size):
                score, new_state = self.compute_single_score(state, label)
                scores[i, label] = score
                new_states[i].append(new_state)

        return scores, new_states

    def compute_single_score(self, state: "kenlm.State", label: int) -> tuple[float, "kenlm.State"]:
        """
        Computes the score with KenLM N-gram language model for `label` given `state`
        Args:
            state: kenLM state
            label: text unit

        Returns:
            tuple: score, next state
        """
        if self.token_offset:
            label = chr(label + self.token_offset)
        else:
            label = str(label)

        next_state = kenlm.State()
        lm_score = self.ngram_lm.BaseScore(state, label, next_state)
        lm_score *= 1.0 / np.log10(np.e)

        return lm_score, next_state

    def compute_sentence_score(self, labels: list[int], bos=True) -> float:
        """
        Compute
        Args:
            labels:
            bos: start with BOS (begin-of-sentence) symbol

        Returns:

        """
        state = self.get_init_state(bos=bos)
        total_score = 0.0
        for label in labels:
            score, state = self.compute_single_score(state=state, label=label)
            total_score += score
        return total_score


class NGram(NamedTuple):
    """Structure (tuple) to represent N-Gram element (symbols, weight, backoff)"""

    weight: float
    backoff: float
    symbols: tuple[int, ...]


class Arc(NamedTuple):
    """Structure (tuple) to represent arc in the weighted acceptor"""

    weight: float
    ilabel: int
    to: int


class FastNGramLM(nn.Module):
    """
    N-Gram LM supporting batched queries. Fast implementation for GPU. Supports autograd (differentiable weights).
    """
    UNK_ID = -3
    BACKOFF_ID = -10
    SPECIAL_SYMBOLS_MAP = {"<s>": -1, "</s>": -2, "<unk>": UNK_ID}
    START_STATE = 0
    BOS_STATE = 1

    def __init__(self, num_states: int, num_arcs: int, vocab_size: int, token_offset=100, use_triton=True):
        super().__init__()
        if not TRITON_AVAILABLE and use_triton:
            logging.warning("Triton not found, falling back to PyTorch")
            use_triton = False
        if not use_triton:
            logging.warning(
                "Triton is disabled. "
                "NB: version without Triton is not compatible with Cuda graphs, decoding can be slow"
            )
        self.use_triton = use_triton

        self.token_offset = token_offset
        self.vocab_size = vocab_size
        self.num_states = num_states
        self.num_arcs = num_arcs
        num_arcs_extended = num_arcs + self.vocab_size  # + extra padding

        # parameters: weights (forward/backoff)
        self.arcs_weights = nn.Parameter(torch.zeros([num_arcs_extended]))
        self.backoff_weights = nn.Parameter(torch.zeros([num_states]))

        # buffers: LM (suffix tree) structure
        self.register_buffer("from_states", torch.zeros([num_arcs_extended], dtype=torch.int64))
        self.register_buffer("to_states", torch.zeros([num_arcs_extended], dtype=torch.int64))
        self.register_buffer("ilabels", torch.zeros([num_arcs_extended], dtype=torch.int64))
        self.register_buffer("backoff_to_states", torch.zeros([num_states], dtype=torch.int64))

        self.register_buffer("state_start_arcs", torch.zeros([num_states], dtype=torch.int64))
        self.register_buffer("state_end_arcs", torch.zeros([num_states], dtype=torch.int64))
        self.register_buffer("state_order", torch.zeros([num_states], dtype=torch.int64))

    @classmethod
    def from_arpa(cls, lm_path: Path | str, vocab_size: int, token_offset=100, use_triton=True) -> "FastNGramLM":
        logging.info(f"FastNGramLM: reading LM {lm_path}")
        ngrams = cls._read_ngrams(lm_path=lm_path, token_offset=token_offset)
        adjacency, num_states = cls._build_suffix_tree(ngrams=ngrams)
        num_arcs = sum(
            len(state_arcs) if state != cls.START_STATE else vocab_size
            for state, state_arcs in enumerate(adjacency)
        )
        # self._suffix_tree_to_torch(adjacency=adjacency)
        model = FastNGramLM(num_states=num_states, num_arcs=num_arcs, vocab_size=vocab_size, token_offset=token_offset, use_triton=use_triton)
        model._init_from_suffix_tree(adjacency=adjacency)
        return model

    @classmethod
    def _read_ngrams(cls, lm_path: Path | str, token_offset: int) -> list[list[NGram]]:
        ngram2cnt_read = defaultdict(int)
        ngram2cnt = defaultdict(int)
        ngrams = []
        special_words_pattern = '|'.join(re.escape(symbol) for symbol in cls.SPECIAL_SYMBOLS_MAP.keys())
        pattern = re.compile(rf'({special_words_pattern}|.)\s?')
        with open(lm_path, "r", encoding="utf-8") as f:
            is_header = True
            cur_order = 0
            for i, line in enumerate(tqdm(f.readlines())):
                if i == 0:
                    assert line.strip() == "\\data\\"
                    continue

                if line.endswith("\n"):
                    line = line[:-1]

                if not line:
                    continue

                if line.startswith("\\end\\"):
                    break

                if is_header:
                    if line.startswith("ngram"):
                        ngram_order, cnt = line.split("=")
                        order = int(ngram_order.split()[-1])
                        cnt = int(cnt)
                        ngram2cnt[order] = cnt
                        continue
                    else:
                        is_header = False
                        max_order = max(ngram2cnt.keys())

                if line.startswith("\\"):
                    cur_order = int(line.split("-")[0][1:])
                    ngrams.append([])
                    continue

                ngrams[-1].append(cls._line_to_ngram(line, pattern=pattern, token_offset=token_offset))
                ngram2cnt_read[cur_order] += 1
            assert ngram2cnt == ngram2cnt_read
            assert len(ngrams) == max_order
            logging.info(f"Loaded model, order={max_order}")
            return ngrams

    @classmethod
    def _line_to_ngram(cls, line: str, pattern: re.Pattern, token_offset: int) -> NGram:
        weight, symbols_str, *backoff_opt = line.split("\t")
        if backoff_opt:
            assert len(backoff_opt) == 1
            backoff = _log_10_to_e(float(backoff_opt[0]))
        else:
            backoff = 0.0
        weight = _log_10_to_e(float(weight))
        symbols_re = pattern.findall(symbols_str)

        symbols = tuple(
            (
                ord(symbol) - token_offset
                if symbol not in cls.SPECIAL_SYMBOLS_MAP
                else cls.SPECIAL_SYMBOLS_MAP[symbol]
            )
            for symbol in symbols_re
        )
        return NGram(weight=weight, backoff=backoff, symbols=symbols)

    @classmethod
    def _build_suffix_tree(cls, ngrams: list[list[NGram]]) -> tuple[list[dict[int, Arc]], int]:
        logging.info("FastNGramLM: Building prefix tree")
        adjacency: list[dict[int, Arc]] = [dict(), dict()]
        num_states = 2  # start, bos
        states_cache = dict()

        for ngram in ngrams[0]:
            assert len(ngram.symbols) == 1
            symbol = ngram.symbols[0]
            if symbol == -1:
                # bos
                adjacency[cls.START_STATE][symbol] = Arc(weight=ngram.weight, ilabel=symbol, to=cls.BOS_STATE)
                adjacency[cls.BOS_STATE][cls.BACKOFF_ID] = Arc(
                    weight=ngram.backoff, ilabel=cls.BACKOFF_ID, to=cls.START_STATE
                )
                states_cache[ngram.symbols] = cls.BOS_STATE
            else:
                assert symbol >= 0 or symbol in {-2, cls.UNK_ID}
                to_state = num_states
                num_states += 1
                adjacency[cls.START_STATE][symbol] = Arc(weight=ngram.weight, ilabel=symbol, to=to_state)
                adjacency.append(
                    {cls.BACKOFF_ID: Arc(weight=ngram.backoff, ilabel=cls.BACKOFF_ID, to=cls.START_STATE)}
                )
                states_cache[ngram.symbols] = to_state

        max_order = len(ngrams)
        for order in tqdm(range(2, max_order + 1)):
            ngram: NGram
            for ngram in ngrams[order - 1]:
                state = states_cache[ngram.symbols[:-1]]
                backoff_state = states_cache[ngram.symbols[1:]]
                symbol = ngram.symbols[-1]
                if order < max_order:
                    to_state = num_states
                    num_states += 1
                    adjacency[state][symbol] = Arc(weight=ngram.weight, ilabel=symbol, to=to_state)
                    adjacency.append(
                        {cls.BACKOFF_ID: Arc(weight=ngram.backoff, ilabel=cls.BACKOFF_ID, to=backoff_state)}
                    )
                    states_cache[ngram.symbols] = to_state
                else:
                    adjacency[state][symbol] = Arc(weight=ngram.weight, ilabel=symbol, to=backoff_state)
        return adjacency, num_states

    def _init_from_suffix_tree(self, adjacency):
        logging.info("Converting suffix tree to PyTorch")
        num_arcs = sum(
            len(state_arcs) if state != self.START_STATE else self.vocab_size
            for state, state_arcs in enumerate(adjacency)
        )

        num_arcs_extended = num_arcs + self.vocab_size  # + extra padding

        # NB: using numpy -> PyTorch, since assigning items directly to PyTorch tensors is extremely slow

        arcs_weights = np.zeros([num_arcs_extended], dtype=np.float32)
        from_states = np.zeros([num_arcs_extended], dtype=np.int64)
        to_states = np.zeros([num_arcs_extended], dtype=np.int64)
        ilabels = np.zeros([num_arcs_extended], dtype=np.int64)

        backoff_weights = np.zeros([self.num_states], dtype=np.float32)
        backoff_to_states = np.zeros([self.num_states], dtype=np.int64)

        state_start_arcs = np.zeros([self.num_states], dtype=np.int64)
        state_end_arcs = np.zeros([self.num_states], dtype=np.int64)
        state_order = np.zeros([self.num_states], dtype=np.int64)

        unk_prob = adjacency[0][self.UNK_ID].weight

        i = 0
        # TODO: arc to start? +1
        state_order[self.START_STATE] = 1
        state_order[self.BOS_STATE] = 2
        for ilabel in range(self.vocab_size):
            if ilabel in adjacency[self.START_STATE]:
                arc = adjacency[self.START_STATE][ilabel]
                arcs_weights[i] = arc.weight
                from_states[i] = self.START_STATE
                to_states[i] = arc.to
                ilabels[i] = arc.ilabel
            else:
                arcs_weights[i] = unk_prob
                from_states[i] = self.START_STATE
                to_states[i] = self.START_STATE
                ilabels[i] = ilabel
            i += 1
        state_end_arcs[self.START_STATE] = i

        for state in tqdm(range(0, self.num_states)):
            if state == self.START_STATE:
                continue
            state_start_arcs[state] = i
            for arc in sorted(adjacency[state].values(), key=lambda arc: arc.ilabel):
                # TODO: batch sort in PyTorch?
                if arc.ilabel >= 0:
                    arcs_weights[i] = arc.weight
                    from_states[i] = state
                    to_states[i] = arc.to
                    ilabels[i] = arc.ilabel
                    i += 1
                elif arc.ilabel == self.BACKOFF_ID:
                    # backoff
                    backoff_weights[state] = arc.weight
                    backoff_to_states[state] = arc.to
                    state_order[state] = state_order[arc.to] + 1
                else:
                    continue
            state_end_arcs[state] = i

        # parameters: weights
        self.arcs_weights.data.copy_(torch.from_numpy(arcs_weights))
        self.backoff_weights.data.copy_(torch.from_numpy(backoff_weights))

        # buffers: LM (suffix tree) structure
        self.from_states.data.copy_(torch.from_numpy(from_states))
        self.to_states.data.copy_(torch.from_numpy(to_states))
        self.ilabels.data.copy_(torch.from_numpy(ilabels))
        self.backoff_to_states.data.copy_(torch.from_numpy(backoff_to_states))

        self.state_start_arcs.data.copy_(torch.from_numpy(state_start_arcs))
        self.state_end_arcs.data.copy_(torch.from_numpy(state_end_arcs))
        self.state_order.data.copy_(torch.from_numpy(state_order))

        assert self.state_order.min().item() == 1
        # assert self.state_order.max().item() == max_order

    # def compute_sentence_score_cpu(self, sentence: list[int], bos=True, verbose=False):
    #     state = self.BOS_STATE if bos else self.START_STATE
    #     weight = 0.0
    #     for token in sentence:
    #         if verbose:
    #             print(f"Token: {token}")
    #         while token not in self.adjacency[state] and state != self.START_STATE:
    #             if verbose:
    #                 print(f"--State: {state}")
    #                 print(f"--Backoff: {self.adjacency[state][self.BACKOFF_ID].weight}")
    #             weight += self.adjacency[state][self.BACKOFF_ID].weight
    #             state = self.adjacency[state][self.BACKOFF_ID].to
    #         if state == self.START_STATE and token not in self.adjacency[state]:
    #             token = self.UNK_ID  # unk
    #         if verbose:
    #             print(f"--Final state: {state}")
    #             print(f"--add-weight: {self.adjacency[state][token].weight:.4f}")
    #         weight += self.adjacency[state][token].weight
    #         state = self.adjacency[state][token].to
    #     return weight

    def compute_sentence_score(self, sentence: list[int], bos=True):
        device = self.arcs_weights.device
        # sentence = torch.tensor(sentence, device=device)[None, :]
        states = self.get_init_states(batch_size=1, bos=bos)
        weight = torch.tensor(0, device=device, dtype=self.arcs_weights.dtype)
        for token in sentence:
            new_scores, new_states_candidates = self.compute_scores_batch(states=states)
            weight += new_scores[0, token]
            states = new_states_candidates[:, token]
        return weight

    @classmethod
    def _log_e_score(cls, score):
        return score / np.log10(np.e)

    def get_init_states(self, batch_size: int, bos=True):
        device = self.arcs_weights.device
        return torch.full(
            [batch_size], fill_value=self.BOS_STATE if bos else self.START_STATE, device=device, dtype=torch.long
        )

    def forward(
        self, labels: torch.Tensor, labels_lengths: torch.Tensor | None = None, bos: bool = True
    ) -> torch.Tensor:
        """
        Compute log-probabilities for all labels in utterances using N-Gram LM.

        Args:
            labels: label sequences [B x L]
            labels_lengths (optional): lengths of the label sequences
            bos: start with BOS symbol

        Returns:
            Tensor [B x L] with scores for each label in the utterance
        """
        device = labels.device
        batch_size, max_length = labels.shape
        if labels_lengths is None:
            labels_lengths = torch.full([batch_size], fill_value=max_length, dtype=torch.int32, device=device)
        scores = torch.zeros(labels.shape, device=device)
        states = self.get_init_states(batch_size=batch_size, bos=bos)
        # TODO(vbataev): faster algorithm
        for i in range(max_length):
            # TODO(vbataev): support differentiable implementation with Triton
            step_scores, states = self._compute_scores_batch_pytorch(states)
            scores[:, i] = step_scores.gather(dim=1, index=labels[:, i]) * (i < labels_lengths)
        return scores

    def compute_scores_batch(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_triton and states.device.type == "cuda":
            return self._compute_scores_batch_triton(states=states)
        return self._compute_scores_batch_pytorch(states=states)

    @triton_required
    def _compute_scores_batch_triton(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = states.shape[0]
        device = states.device
        scores = torch.empty([batch_size, self.vocab_size], device=device, dtype=self.arcs_weights.dtype)
        new_states = torch.empty([batch_size, self.vocab_size], dtype=torch.long, device=device)

        NUM_BLOCKS = batch_size
        BLOCK_SIZE = triton.next_power_of_2(self.vocab_size)

        _ngram_triton_kernel[NUM_BLOCKS,](
            vocab_size=self.vocab_size,
            states_ptr=states,
            new_states_ptr=new_states,
            scores_ptr=scores,
            start_state=self.START_STATE,
            max_order=self.max_order,
            backoff_to_states_ptr=self.backoff_to_states,
            backoff_weights_ptr=self.backoff_weights,
            state_start_arcs_ptr=self.state_start_arcs,
            state_end_arcs_ptr=self.state_end_arcs,
            to_states_ptr=self.to_states,
            ilabels_ptr=self.ilabels,
            arcs_weights_ptr=self.arcs_weights,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return scores, new_states

    def _compute_scores_batch_pytorch(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = states.shape[0]
        device = states.device
        current_states = states.clone()

        out_scores = torch.zeros(batch_size, self.vocab_size, device=device)
        out_states = torch.full([batch_size, self.vocab_size], fill_value=-1, dtype=torch.long, device=device)
        state_found = torch.full([batch_size, self.vocab_size], fill_value=False, dtype=torch.bool, device=device)

        all_labels = torch.arange(self.vocab_size, device=device)
        batch_indices = torch.arange(batch_size, device=device)

        lm_not_done = torch.full([batch_size], fill_value=True, dtype=torch.bool, device=device)
        # max_iter = self.state_order[states].max().item()
        # for _ in range(max_iter):
        while lm_not_done.any():
            start = self.state_start_arcs[current_states]
            indices = start[:, None] + all_labels[None, :]
            end = self.state_end_arcs[current_states]
            mask = indices < end[:, None]
            mask &= lm_not_done[:, None]
            mask_flat = mask.view(-1)
            indices_flat = indices.view(-1)

            # scores_add = torch.zeros_like(scores)
            # new_states_add = torch.full_like(new_states, fill_value=-1)
            # scores_add[batch_indices.repeat_interleave(self.vocab_size)[mask_flat], \
            # self.ilabels[indices_flat][mask_flat]] = self.arcs_weights[indices_flat][mask_flat]
            # new_states_add[batch_indices.repeat_interleave(self.vocab_size)[mask_flat], \
            # self.ilabels[indices_flat][mask_flat]] = self.to_states[indices_flat][mask_flat]
            scores_add = torch.zeros([batch_size, self.vocab_size + 1], device=device, dtype=out_scores.dtype)
            out_states_add = torch.full(
                [batch_size, self.vocab_size + 1], fill_value=-1, device=device, dtype=torch.long
            )
            ilabels = self.ilabels[indices_flat] * mask_flat + ~mask_flat * self.vocab_size
            # todo: is this UB or not?
            scores_add[batch_indices.repeat_interleave(self.vocab_size), ilabels] = self.arcs_weights[indices_flat]
            out_states_add[batch_indices.repeat_interleave(self.vocab_size), ilabels] = self.to_states[indices_flat]

            # scores[~state_found] += scores_add[~state_found]
            torch.where(state_found, out_scores, out_scores + scores_add[:, : self.vocab_size], out=out_scores)
            # new_states[~state_found] = new_states_add[~state_found]
            torch.where(state_found, out_states, out_states_add[:, : self.vocab_size], out=out_states)
            state_found = out_states != -1
            lm_not_done &= current_states != self.START_STATE
            out_scores += self.backoff_weights[current_states][:, None] * (~state_found)
            torch.where(lm_not_done, self.backoff_to_states[current_states], current_states, out=current_states)
        return out_scores, out_states
