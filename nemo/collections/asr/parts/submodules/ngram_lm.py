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
from collections.abc import Iterator
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import List, NamedTuple, Optional, cast

import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from omegaconf import MISSING, DictConfig, OmegaConf
from tqdm.auto import tqdm

from nemo.collections.common.parts import NEG_INF
from nemo.core import ModelPT, PretrainedModelInfo
from nemo.core.utils.optional_libs import KENLM_AVAILABLE, TRITON_AVAILABLE, kenlm_required, triton_required
from nemo.utils import logging

if KENLM_AVAILABLE:
    import kenlm

if TRITON_AVAILABLE:
    import triton

    from nemo.collections.asr.parts.submodules.ngram_lm_triton import _ngram_advance_triton_kernel


def _log_10_to_e(score):
    return score / np.log10(np.e)


try:
    from numba import njit

    NUMBA_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:

    @njit
    def _find_state_numba(to_states, ilabels, state_start_arcs, state_end_arcs, symbols, bos_id) -> int:
        assert len(symbols) >= 1
        label = symbols[0]
        if label == bos_id:
            state = 1
        elif label >= 0:
            state = to_states[label]
            assert ilabels[label] == label
        else:
            raise NotImplementedError

        for label in symbols[1:]:
            arc_id = (
                np.searchsorted(ilabels[state_start_arcs[state] : state_end_arcs[state]], label, side='left')
                + state_start_arcs[state]
            )
            assert ilabels[arc_id] == label
            state = to_states[arc_id]
        return state


class KenLMWrapper:
    """
    KenLM model wrapper for single element and batched queries for decoding (reference) and testing purposes.
    """

    @kenlm_required
    def __init__(self, lm_path: Path | str, vocab_size: int, token_offset: int = 100):
        self.ngram_lm = kenlm.Model(str(lm_path))
        self.token_offset = token_offset
        self.vocab_size = vocab_size

    def get_init_state(self, bos=True):
        init_lm_state = kenlm.State()

        if not bos:
            return init_lm_state

        self.ngram_lm.BeginSentenceWrite(init_lm_state)
        return init_lm_state

    def get_init_states(self, batch_size: int, bos=True):
        return [self.get_init_state(bos=bos) for _ in range(batch_size)]

    def advance(self, states: list["kenlm.State"]) -> tuple[torch.Tensor, list[list["kenlm.State"]]]:
        batch_size = len(states)
        new_states = [[] for _ in range(len(states))]
        scores = torch.zeros(batch_size, self.vocab_size)
        for i, state in enumerate(states):
            for label in range(self.vocab_size):
                score, new_state = self.advance_single(state, label)
                scores[i, label] = score
                new_states[i].append(new_state)

        return scores, new_states

    def advance_single(self, state: "kenlm.State", label: int) -> tuple[float, "kenlm.State"]:
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

    def score_sentence(self, sentence: list[int], bos=True) -> torch.Tensor:
        """
        Compute
        Args:
            labels:
            bos: start with BOS (begin-of-sentence) symbol

        Returns:

        """
        state = self.get_init_state(bos=bos)
        scores = []
        for label in sentence:
            score, state = self.advance_single(state=state, label=label)
            scores.append(score)
        return torch.FloatTensor(scores)

    def score_sentences(self, sentences: list[list[int]], bos=True) -> torch.Tensor:
        return torch.stack([self.score_sentence(sentence, bos=bos) for sentence in sentences], dim=0)


class NGram(NamedTuple):
    """Structure (tuple) to represent N-Gram element (symbols, weight, backoff)"""

    symbols: tuple[int, ...]
    weight: float
    backoff: float


class Arc(NamedTuple):
    """Structure (tuple) to represent arc in the weighted acceptor"""

    weight: float
    ilabel: int
    to: int


_EOS_ID = -2


@dataclass
class SuffixTreeStorage:
    num_states_max: InitVar[int]
    num_arcs_max: InitVar[int]

    vocab_size: int
    max_order: int

    arcs: np.ndarray = field(init=False)
    states: np.ndarray = field(init=False)

    _arc_cache: dict[tuple[int, ...], int] = field(default_factory=dict)

    unk_prob: float = float("-inf")
    normalize_unk: bool = True

    num_states: int = 0
    num_arcs: int = 0

    def __post_init__(self, num_states_max: int, num_arcs_max: int):
        if max(num_states_max, num_arcs_max) < np.iinfo(np.int32).max:
            int_np_dtype = np.int32
        else:
            int_np_dtype = np.int64
        self.arcs = np.zeros(
            [num_arcs_max],
            dtype=[("from", int_np_dtype), ("to", int_np_dtype), ("ilabel", int_np_dtype), ("weight", np.float32)],
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

    def _add_unigrams(self, ngrams: np.ndarray, bos_id: int, unk_id: int):
        start_state = 0
        bos_state = 1
        assert bos_id < 0 and unk_id < 0
        bos_unigram = None

        ngrams.sort(order="symbols")
        for ngram in ngrams:
            assert len(ngram["symbols"]) == 1  # unigrams
            symbol = ngram["symbols"][-1]
            if symbol == unk_id:
                self.unk_prob = ngram["weight"]
            elif symbol == bos_id:
                bos_unigram = ngram
        assert bos_unigram is not None

        self.num_states = 2  # SOS + BOS
        self.num_arcs = 0
        # TODO: order 1 or 0?
        # state: start_arcs, end_arcs, order, backoff_to, backoff_weight
        self.states[start_state] = (0, self.vocab_size, 1, start_state, 0.0, NEG_INF)
        added_symbols = set()
        num_vocab_labels = 0
        for ngram in ngrams:
            ilabel = ngram["symbols"][-1]
            if ilabel < 0:
                # special symbol
                if ilabel == _EOS_ID:
                    self.states[start_state]["final"] = ngram["weight"]
                continue
            assert ilabel < self.vocab_size
            arc_id = ilabel
            added_symbols.add(ilabel)
            next_state = self.num_states
            self.num_states += 1
            self.arcs[arc_id] = (start_state, next_state, ilabel, ngram["weight"])
            self.num_arcs += 1
            # state order
            self.states[next_state] = (
                0,
                0,
                self.states[start_state]["order"] + 1,
                start_state,
                ngram["backoff"],
                NEG_INF,
            )
            num_vocab_labels += 1

        if self.normalize_unk:
            num_unk_labels = self.vocab_size - num_vocab_labels
            self.unk_prob -= np.log(num_unk_labels)
        for ilabel in range(self.vocab_size):
            if ilabel not in added_symbols:
                self.arcs[ilabel] = (start_state, start_state, ilabel, self.unk_prob)
                self.num_arcs += 1

        # add BOS unigram
        # NB: we do not add BOS unigram to the arcs, but only to the states
        self.states[bos_state] = (
            0,
            0,
            self.states[start_state]["order"] + 1,
            start_state,
            bos_unigram["backoff"],
            NEG_INF,
        )

    def find_state(self, symbols: tuple[int, ...], bos_id: int) -> int:
        if len(symbols) > 1:
            return self._arc_cache[tuple(symbols)]
        assert len(symbols) == 1
        label = symbols[0]
        if label == bos_id:
            return 1
        elif label >= 0:
            return self.arcs[label]["to"]
        raise ValueError(f"Invalid symbol {label}")

    def _add_ngrams_next_order(self, ngrams: np.ndarray, bos_id: int):
        ngrams.sort(order="symbols")
        new_arc_cache = dict()
        for ngram in tqdm(ngrams):
            symbols = ngram["symbols"].item()
            ilabel = symbols[-1]
            from_state = self.find_state(symbols[:-1], bos_id=bos_id)
            if ilabel < 0:
                assert ilabel == -2
                self.states[from_state]["final"] = ngram["weight"]
                continue
            assert 0 <= ilabel < self.vocab_size
            backoff_state = self.find_state(symbols[1:], bos_id=bos_id)

            arc_id = self.num_arcs
            next_state = self.num_states
            self.num_arcs += 1
            self.num_states += 1
            self.arcs[arc_id] = (from_state, next_state, ilabel, ngram["weight"])
            # state: start_arcs, end_arcs, order, backoff_to, backoff_weight
            self.states[next_state] = (
                0,
                0,
                self.states[from_state]["order"] + 1,
                backoff_state,
                ngram["backoff"],
                NEG_INF,
            )

            if self.states[from_state]["arcs_start"] == 0:
                self.states[from_state]["arcs_start"] = arc_id
                self.states[from_state]["arcs_end"] = arc_id + 1
            else:
                assert self.states[from_state]["arcs_end"] == arc_id
                self.states[from_state]["arcs_end"] = arc_id + 1
            # cache state
            new_arc_cache[symbols] = next_state
        self._arc_cache = new_arc_cache

    def _start_adding_ngrams_for_order(self, order: int, max_ngrams: int):
        self._start_arcs = self.num_arcs
        self._cur_order = order
        if order < self.max_order:
            dtype = [
                ("symbols", [(f"{i}", np.int32) for i in range(order)]),
                ("weight", np.float32),
                ("backoff", np.float32),
            ]
            self._ngrams = np.zeros([max_ngrams], dtype=dtype)
            self._ngrams_cnt = 0
        # for max order - no need in accumulator

    def _add_ngram(self, ngram: NGram, bos_id: int):
        assert len(ngram.symbols) == self._cur_order
        if self._cur_order == self.max_order:
            self._add_ngram_max_order(ngram=ngram, bos_id=bos_id)
            return
        self._ngrams[self._ngrams_cnt] = (ngram.symbols, ngram.weight, ngram.backoff)
        self._ngrams_cnt += 1

    def _end_adding_ngrams_for_order(self, order: int, bos_id: int, unk_id: int):
        if order == 1:
            assert self._ngrams.shape[0] == self._ngrams_cnt
            self._add_unigrams(ngrams=self._ngrams, bos_id=bos_id, unk_id=unk_id)
            self._ngrams = None
            self._ngrams_cnt = 0
        elif order < self.max_order:
            assert self._ngrams.shape[0] == self._ngrams_cnt
            self._add_ngrams_next_order(ngrams=self._ngrams, bos_id=bos_id)
            self._ngrams = None
            self._ngrams_cnt = 0
        else:
            self._end_adding_ngrams_max_order()

    def _add_ngram_max_order(self, ngram: NGram, bos_id: int):
        ilabel = ngram.symbols[-1]
        from_state = self.find_state(ngram.symbols[:-1], bos_id=bos_id)
        if ilabel < 0:
            assert ilabel == -2
            self.states[from_state]["final"] = ngram.weight
            return
        backoff_state = self.find_state(ngram.symbols[1:], bos_id=bos_id)

        arc_id = self.num_arcs
        self.num_arcs += 1
        self.arcs[arc_id] = (from_state, backoff_state, ilabel, ngram.weight)

    def _end_adding_ngrams_max_order(self):
        self.arcs[self._start_arcs : self.num_arcs].sort(order=["from", "ilabel"])
        for arc_i in range(self._start_arcs, self.num_arcs):
            from_state = self.arcs[arc_i]["from"]
            if self.states[from_state]["arcs_start"] == 0:
                self.states[from_state]["arcs_start"] = arc_i
            self.states[from_state]["arcs_end"] = arc_i + 1

    def sanity_check(self):
        assert (self.arcs["ilabel"][: self.num_arcs] < self.vocab_size).all()
        assert (self.arcs["ilabel"][: self.num_arcs] >= 0).all()


@dataclass
class NGramLMConfig:
    """
    N-Gram LM Config
    """

    num_states: int = MISSING
    num_arcs: int = MISSING
    max_order: int = MISSING
    vocab_size: int = MISSING
    use_triton: bool | None = None


class FastNGramLM(ModelPT):
    """
    N-Gram LM supporting batched queries. Fast implementation for parallel queries for full vocabulary.
    Supports autograd (differentiable weights).
    """

    UNK_ID = -3
    BACKOFF_ID = -10
    BOS_ID = -1  # Begin-of-Sentence
    EOS_ID = _EOS_ID  # End-of-Sentence
    SPECIAL_SYMBOLS_MAP = {"<s>": BOS_ID, "</s>": EOS_ID, "<unk>": UNK_ID}
    START_STATE = 0
    BOS_STATE = 1

    def __init__(
        self,
        cfg: DictConfig,
        trainer: Trainer = None,
        # num_states: int, num_arcs: int, max_order: int, vocab_size: int, use_triton: bool | None = None
    ):
        """
        Stubs for constructor that does not initialize the structure.
        This constructor can be useful when storing/loading module using native torch serialization mechanism
        instead of directly reading ARPA model -> converting to Torch, which can be slow for large N-Gram models
        (of several GBs).

        Args:
            num_states: number of states in graph
            num_arcs: number of arcs (transitions) in graph
            max_order: maximum order of n-gram LM (maximum possible nubmer of transitions without backoffs)
            vocab_size: vocabulary size (existing vocabulary units in LM; should not include blank etc.)
            use_triton: allow using Triton implementation;
                None (default) means "auto" (used if available), True means forced mode
                (will crash if Triton is unavailable)
        """
        # cfg = OmegaConf.merge(OmegaConf.structured(NGramLMConfig), cfg)
        super().__init__(cfg=cfg, trainer=trainer)
        cfg = cast(NGramLMConfig, cfg)
        self.use_triton = cfg.use_triton if cfg.use_triton is not None else TRITON_AVAILABLE
        if not self.use_triton:
            logging.warning(
                "Triton is disabled. Version without Triton is not compatible with Cuda graphs; decoding can be slow"
            )

        self.vocab_size = cfg.vocab_size
        self.num_states = cfg.num_states
        self.num_arcs = cfg.num_arcs
        self.max_order = cfg.max_order
        self.num_arcs_extended = cfg.num_arcs + self.vocab_size  # + extra padding

        # parameters: weights (forward/backoff)
        self.arcs_weights = nn.Parameter(torch.zeros([self.num_arcs_extended]))
        self.backoff_weights = nn.Parameter(torch.zeros([self.num_states]))
        self.final_weights = nn.Parameter(torch.zeros([self.num_states]))

        if max(self.num_states, self.num_arcs_extended) < torch.iinfo(torch.int32).max:
            int_dtype = torch.int32
        else:
            int_dtype = torch.int64
        # buffers: LM (suffix tree) structure
        self.register_buffer("from_states", torch.zeros([self.num_arcs_extended], dtype=int_dtype))
        self.register_buffer("to_states", torch.zeros([self.num_arcs_extended], dtype=int_dtype))
        self.register_buffer("ilabels", torch.zeros([self.num_arcs_extended], dtype=int_dtype))
        self.register_buffer("backoff_to_states", torch.zeros([self.num_states], dtype=int_dtype))

        self.register_buffer("state_start_arcs", torch.zeros([self.num_states], dtype=int_dtype))
        self.register_buffer("state_end_arcs", torch.zeros([self.num_states], dtype=int_dtype))
        self.register_buffer("state_order", torch.zeros([self.num_states], dtype=int_dtype))

        self._final_resolved = False

    def list_available_models(cls):
        return []

    def setup_training_data(self, train_data_config: DictConfig | dict):
        pass

    def setup_validation_data(self, val_data_config: DictConfig | dict):
        pass

    @classmethod
    def from_nemo(
        cls,
        lm_path: Path | str,
        vocab_size: int,
        use_triton: bool | None = None,
    ) -> "FastNGramLM":
        """
        Constructor from Nemo checkpoint (state dict).

        Args:
            path: path to .nemo checkpoint
        """
        model = FastNGramLM.restore_from(restore_path=str(lm_path), map_location="cpu")
        model.resolve_final()
        assert model.vocab_size == vocab_size
        model.use_triton = use_triton if use_triton is not None else TRITON_AVAILABLE
        if not model.use_triton:
            logging.warning(
                "Triton is disabled. Version without Triton is not compatible with Cuda graphs; decoding can be slow"
            )
        return model

    @classmethod
    def from_file(
        cls,
        lm_path: Path | str,
        vocab_size: int,
        normalize_unk: bool = True,
        token_offset: int = 100,
        use_triton: bool | None = None,
    ) -> "FastNGramLM":
        if not isinstance(lm_path, Path):
            lm_path = Path(lm_path)
        if lm_path.suffix == ".nemo":
            return cls.from_nemo(lm_path=lm_path, vocab_size=vocab_size, use_triton=use_triton)
        return cls.from_arpa(
            lm_path=lm_path,
            vocab_size=vocab_size,
            normalize_unk=normalize_unk,
            token_offset=token_offset,
            use_triton=use_triton,
        )

    @classmethod
    def from_arpa(
        cls,
        lm_path: Path | str,
        vocab_size: int,
        normalize_unk: bool = True,
        token_offset: int = 100,
        use_triton: bool | None = None,
    ) -> "FastNGramLM":
        """
        Constructor from ARPA LM (text format).

        Args:
            lm_path: path to ARPA model (human-readable)
            vocab_size: vocabulary size (existing vocabulary units in LM; should not include blank etc.)
            normalize_unk: unk normalization to make all output probabilities sum to 1.0 (default: True).
                Setting to False can be useful for one-to-one comparison with KenLM (tests, etc.).
            token_offset: offset for the tokens used for building ARPA LM
            use_triton: allow using Triton implementation;
                None (default) means "auto" (used if available), True means forced mode
                (will crash if Triton is unavailable)

        Returns:
            FastNGramLM module
        """
        logging.info(f"{cls.__name__}: reading LM from {lm_path}")
        # adjacency: list[dict[int, Arc]] = [dict(), dict()]
        # states_cache = dict()
        with open(lm_path, "r", encoding="utf-8") as f:
            order2cnt = cls._read_header(f=f)
            max_order = max(order2cnt.keys())
            total_ngrams = sum(order2cnt.values())
            max_states = 2 + vocab_size + sum(order2cnt[o] for o in range(2, max_order))  # without last!
            suffix_tree_np = SuffixTreeStorage(
                num_states_max=max_states,
                num_states=0,
                num_arcs=0,
                num_arcs_max=total_ngrams + vocab_size * 2 + 1,
                normalize_unk=normalize_unk,
                vocab_size=vocab_size,
                max_order=max_order,
            )
            ngram_cur_order_i = 0
            cur_order = 1
            for ngram in tqdm(cls._read_ngrams(f=f, token_offset=token_offset), total=total_ngrams):
                if ngram_cur_order_i == 0:
                    suffix_tree_np._start_adding_ngrams_for_order(order=cur_order, max_ngrams=order2cnt[cur_order])
                ngram_cur_order_i += 1
                suffix_tree_np._add_ngram(ngram=ngram, bos_id=cls.BOS_ID)

                if ngram_cur_order_i == order2cnt[cur_order]:
                    suffix_tree_np._end_adding_ngrams_for_order(order=cur_order, bos_id=cls.BOS_ID, unk_id=cls.UNK_ID)
                    logging.info(f"Processed {order2cnt[cur_order]} n-grams of order {cur_order}")
                    cur_order += 1
                    ngram_cur_order_i = 0

                # cls._build_suffix_tree_iterative(
                #     adjacency=adjacency, ngram=ngram, max_order=max_order, states_cache=states_cache
                # )
            assert ngram_cur_order_i == 0
            suffix_tree_np.sanity_check()
        return FastNGramLM.from_suffix_tree(suffix_tree_np=suffix_tree_np, use_triton=use_triton)

    @classmethod
    def from_suffix_tree(cls, suffix_tree_np: SuffixTreeStorage, use_triton: bool | None = None) -> "FastNGramLM":
        model = FastNGramLM(
            OmegaConf.structured(
                NGramLMConfig(
                    num_states=suffix_tree_np.num_states,
                    num_arcs=suffix_tree_np.num_arcs,
                    max_order=suffix_tree_np.max_order,
                    vocab_size=suffix_tree_np.vocab_size,
                    use_triton=use_triton,
                )
            )
        )
        model._init_from_suffix_tree_np(suffix_tree_np=suffix_tree_np)
        model.resolve_final()
        return model

    @classmethod
    def _read_header(cls, f) -> dict[int, int]:
        is_start = True
        order2cnt: dict[int, int] = defaultdict(int)
        for line in f:
            line = line.strip()
            if is_start:
                assert line == "\\data\\"
                is_start = False
                continue

            if line.startswith("ngram"):
                ngram_order, cnt = line.split("=")
                order = int(ngram_order.split()[-1])
                cnt = int(cnt)
                order2cnt[order] = cnt
                continue
            else:
                assert not line, "empty line expected after header"
                break
        return order2cnt

    @classmethod
    def _read_ngrams(cls, f, token_offset: int) -> Iterator[NGram]:
        special_words_pattern = '|'.join(re.escape(symbol) for symbol in cls.SPECIAL_SYMBOLS_MAP.keys())
        pattern = re.compile(rf'({special_words_pattern}|.)\s?')
        for line in f:
            if line.endswith("\n"):
                line = line[:-1]

            if not line:
                continue

            if line.startswith("\\end\\"):
                break

            if line.startswith("\\"):
                # cur_order = int(line.split("-")[0][1:])
                # ngrams.append([])
                continue

            ngram = cls._line_to_ngram(
                special_symbols_map=cls.SPECIAL_SYMBOLS_MAP, line=line, pattern=pattern, token_offset=token_offset
            )
            yield ngram

    @staticmethod
    def _line_to_ngram(
        special_symbols_map: dict[str, int], line: str, pattern: re.Pattern, token_offset: int
    ) -> NGram:
        weight, symbols_str, *backoff_opt = line.split("\t")
        if backoff_opt:
            assert len(backoff_opt) == 1
            backoff = _log_10_to_e(float(backoff_opt[0]))
        else:
            backoff = 0.0
        weight = _log_10_to_e(float(weight))
        symbols_re = pattern.findall(symbols_str)

        symbols = tuple(
            (ord(symbol) - token_offset if symbol not in special_symbols_map else special_symbols_map[symbol])
            for symbol in symbols_re
        )
        return NGram(symbols=symbols, weight=weight, backoff=backoff)

    @classmethod
    def _build_suffix_tree_iterative(
        cls, adjacency: list[dict[int, Arc]], ngram: NGram, max_order: int, states_cache: dict[tuple[int, ...], int]
    ):
        # adjacency: list[dict[int, Arc]] = [dict(), dict()]
        # num_states = 2  # start, bos
        order = len(ngram.symbols)
        symbol = ngram.symbols[-1]
        if order == 1:
            if symbol == cls.BOS_ID:
                # bos
                adjacency[cls.START_STATE][symbol] = Arc(weight=ngram.weight, ilabel=symbol, to=cls.BOS_STATE)
                adjacency[cls.BOS_STATE][cls.BACKOFF_ID] = Arc(
                    weight=ngram.backoff, ilabel=cls.BACKOFF_ID, to=cls.START_STATE
                )
                states_cache[ngram.symbols] = cls.BOS_STATE
            else:
                assert symbol >= 0 or symbol in {cls.EOS_ID, cls.UNK_ID}
                to_state = len(adjacency)
                # num_states += 1
                adjacency[cls.START_STATE][symbol] = Arc(weight=ngram.weight, ilabel=symbol, to=to_state)
                adjacency.append(
                    {cls.BACKOFF_ID: Arc(weight=ngram.backoff, ilabel=cls.BACKOFF_ID, to=cls.START_STATE)}
                )
                states_cache[ngram.symbols] = to_state
        else:
            state = states_cache[ngram.symbols[:-1]]
            backoff_state = states_cache[ngram.symbols[1:]]
            if order < max_order:
                to_state = len(adjacency)
                # num_states += 1
                adjacency[state][symbol] = Arc(weight=ngram.weight, ilabel=symbol, to=to_state)
                adjacency.append({cls.BACKOFF_ID: Arc(weight=ngram.backoff, ilabel=cls.BACKOFF_ID, to=backoff_state)})
                states_cache[ngram.symbols] = to_state
            else:
                adjacency[state][symbol] = Arc(weight=ngram.weight, ilabel=symbol, to=backoff_state)

    def _init_from_suffix_tree(self, adjacency: list[dict[int, Arc]]):
        logging.info("Converting suffix tree to PyTorch")
        num_arcs = sum(
            len(state_arcs) if state != self.START_STATE else self.vocab_size
            for state, state_arcs in enumerate(adjacency)
        )

        num_arcs_extended = num_arcs + self.vocab_size  # + extra padding
        assert self.num_states == len(adjacency)
        assert self.num_arcs == num_arcs
        suffix_tree_np = SuffixTreeStorage(
            num_states_max=self.num_states,
            num_states=self.num_states,
            num_arcs=num_arcs_extended,
            num_arcs_max=num_arcs_extended,
            vocab_size=self.vocab_size,
            max_order=self.max_order,
        )
        suffix_tree_np.unk_prob = adjacency[0][self.UNK_ID].weight

        i = 0
        # TODO: arc to start? +1
        suffix_tree_np.state_order[self.START_STATE] = 1
        suffix_tree_np.state_order[self.BOS_STATE] = 2
        for ilabel in range(self.vocab_size):
            if ilabel in adjacency[self.START_STATE]:
                arc = adjacency[self.START_STATE][ilabel]
                suffix_tree_np.arcs_weights[i] = arc.weight
                suffix_tree_np.from_states[i] = self.START_STATE
                suffix_tree_np.to_states[i] = arc.to
                suffix_tree_np.ilabels[i] = arc.ilabel
            else:
                suffix_tree_np.arcs_weights[i] = suffix_tree_np.unk_prob
                suffix_tree_np.from_states[i] = self.START_STATE
                suffix_tree_np.to_states[i] = self.START_STATE
                suffix_tree_np.ilabels[i] = ilabel
            i += 1
        suffix_tree_np.state_end_arcs[self.START_STATE] = i

        for state in tqdm(range(0, self.num_states)):
            if state == self.START_STATE:
                continue
            suffix_tree_np.state_start_arcs[state] = i
            for arc in sorted(adjacency[state].values(), key=lambda arc: arc.ilabel):
                if arc.ilabel >= 0:
                    suffix_tree_np.arcs_weights[i] = arc.weight
                    suffix_tree_np.from_states[i] = state
                    suffix_tree_np.to_states[i] = arc.to
                    suffix_tree_np.ilabels[i] = arc.ilabel
                    i += 1
                elif arc.ilabel == self.BACKOFF_ID:
                    # backoff
                    suffix_tree_np.backoff_weights[state] = arc.weight
                    suffix_tree_np.backoff_to_states[state] = arc.to
                    suffix_tree_np.state_order[state] = suffix_tree_np.state_order[arc.to] + 1
                else:
                    continue
            suffix_tree_np.state_end_arcs[state] = i

        self._init_from_suffix_tree_np(suffix_tree_np=suffix_tree_np)

    def _init_from_suffix_tree_np(self, suffix_tree_np: SuffixTreeStorage):
        # parameters: weights
        self.arcs_weights.data.copy_(torch.from_numpy(suffix_tree_np.arcs["weight"][: self.num_arcs_extended]))
        self.backoff_weights.data.copy_(torch.from_numpy(suffix_tree_np.states["backoff_w"][: self.num_states]))
        self.final_weights.data.copy_(torch.from_numpy(suffix_tree_np.states["final"][: self.num_states]))

        # buffers: LM (suffix tree) structure
        self.from_states.data.copy_(torch.from_numpy(suffix_tree_np.arcs["from"][: self.num_arcs_extended]))
        self.to_states.data.copy_(torch.from_numpy(suffix_tree_np.arcs["to"][: self.num_arcs_extended]))
        self.ilabels.data.copy_(torch.from_numpy(suffix_tree_np.arcs["ilabel"][: self.num_arcs_extended]))
        self.backoff_to_states.data.copy_(torch.from_numpy(suffix_tree_np.states["backoff_to"][: self.num_states]))

        self.state_start_arcs.data.copy_(torch.from_numpy(suffix_tree_np.states["arcs_start"][: self.num_states]))
        self.state_end_arcs.data.copy_(torch.from_numpy(suffix_tree_np.states["arcs_end"][: self.num_states]))
        self.state_order.data.copy_(torch.from_numpy(suffix_tree_np.states["order"][: self.num_states]))

        # sanity check
        assert self.state_order.min().item() == 1
        assert self.state_order.max().item() == self.max_order

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
        return self.score_sentences(labels=labels, labels_lengths=labels_lengths, bos=bos)

    def score_sentences(
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
            step_scores, states = self._advance_pytorch(states)
            scores[:, i] = step_scores.gather(dim=1, index=labels[:, i].unsqueeze(0)).squeeze(0) * (i < labels_lengths)
            states = states.gather(dim=1, index=labels[:, i].unsqueeze(0)).squeeze(0)
        return scores

    def advance(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance `states` [B]: return next states [B, V] and scores [B, V] for full vocab
        Args:
            states: batch of states

        Returns:
            tuple with next states and scores
        """
        if self.use_triton and states.device.type == "cuda":
            return self._advance_triton(states=states)
        return self._advance_pytorch(states=states)

    @triton_required
    def _advance_triton(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = states.shape[0]
        device = states.device
        scores = torch.empty([batch_size, self.vocab_size], device=device, dtype=self.arcs_weights.dtype)
        new_states = torch.empty([batch_size, self.vocab_size], dtype=torch.long, device=device)

        _ngram_advance_triton_kernel[batch_size,](
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
            BLOCK_SIZE=triton.next_power_of_2(self.vocab_size),
        )

        return scores, new_states

    def get_final(self, states: torch.Tensor) -> torch.Tensor:
        # TODO: add Triton kernel
        if self._final_resolved:
            return self.final_weights[states]
        logging.warning("Final weights are not resolved; using slow implementation")
        return self._get_final_pytorch(states=states)

    def resolve_final(self):
        if self._final_resolved:
            return
        with torch.no_grad():
            self.final_weights.data.copy_(
                self._get_final_pytorch(states=torch.arange(self.num_states, device=self.final_weights.device))
            )
        self._final_resolved = True

    def _get_final_pytorch(self, states: torch.Tensor) -> torch.Tensor:
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

    def _advance_pytorch(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = states.shape[0]
        device = states.device
        current_states = states.clone()
        states_dtype = current_states.dtype

        out_scores = torch.zeros(batch_size, self.vocab_size, device=device)
        out_states = torch.full([batch_size, self.vocab_size], fill_value=-1, dtype=states_dtype, device=device)
        state_found = torch.full([batch_size, self.vocab_size], fill_value=False, dtype=torch.bool, device=device)
        accumulated_backoff = torch.zeros(batch_size, device=device)

        all_labels = torch.arange(self.vocab_size, device=device)
        batch_indices = torch.arange(batch_size, device=device)

        lm_not_done = torch.full([batch_size], fill_value=True, dtype=torch.bool, device=device)
        num_iterations = 0
        while lm_not_done.any():
            assert num_iterations <= self.max_order, "Infinite loop in LM advance"
            num_iterations += 1
            start = self.state_start_arcs[current_states]
            indices = start[:, None] + all_labels[None, :]
            end = self.state_end_arcs[current_states]
            mask = indices < end[:, None]
            mask &= lm_not_done[:, None]
            mask_flat = mask.view(-1)
            indices_flat = indices.view(-1)
            scores_add = torch.zeros([batch_size, self.vocab_size + 1], device=device, dtype=out_scores.dtype)
            out_states_add = torch.full(
                [batch_size, self.vocab_size + 1], fill_value=-1, device=device, dtype=states_dtype
            )
            ilabels = self.ilabels[indices_flat] * mask_flat + ~mask_flat * self.vocab_size
            scores_add[batch_indices.repeat_interleave(self.vocab_size), ilabels] = self.arcs_weights[indices_flat]
            out_states_add[batch_indices.repeat_interleave(self.vocab_size), ilabels] = self.to_states[
                indices_flat
            ].to(states_dtype)
            out_scores = torch.where(
                state_found, out_scores, accumulated_backoff.unsqueeze(-1) + scores_add[:, : self.vocab_size]
            )
            out_states = torch.where(state_found, out_states, out_states_add[:, : self.vocab_size])
            state_found = out_states != -1
            lm_not_done &= current_states != self.START_STATE
            accumulated_backoff += self.backoff_weights[current_states] * lm_not_done
            torch.where(lm_not_done, self.backoff_to_states[current_states], current_states, out=current_states)
        return out_scores, out_states
