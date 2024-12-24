import numpy as np
from typing import Tuple
import torch
from collections import defaultdict
from typing import NamedTuple
import torch.nn as nn
from tqdm.auto import tqdm
from nemo.utils import logging
from pathlib import Path
import re
import random

try:
    import kenlm

    HAVE_KENLM = True
except (ModuleNotFoundError, ImportError):
    HAVE_KENLM = False

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except (ModuleNotFoundError, ImportError):
    HAVE_TRITON = False


def _log_e_score(score):
    return score / np.log10(np.e)

if HAVE_TRITON:
    @triton.jit
    def _ngram_triton_kernel(
        vocab_size: "tl.constexpr",
        states_ptr,
        new_states_ptr,
        scores_ptr,
        start_state: int,
        max_order: int,
        backoff_to_states_ptr,
        backoff_weights_ptr,
        state_start_arcs_ptr,
        state_end_arcs_ptr,
        to_states_ptr,
        ilabels_ptr,
        arcs_weights_ptr,
        BLOCK_SIZE: "tl.constexpr",
    ):
        batch_i = tl.program_id(0)
        cur_state = tl.load(states_ptr + batch_i)

        vocab_offsets = tl.arange(0, BLOCK_SIZE)
        vocab_mask = vocab_offsets < vocab_size
        tl.store(new_states_ptr + batch_i * vocab_size + vocab_offsets, -1, mask=vocab_mask)
        tl.store(scores_ptr + batch_i * vocab_size + vocab_offsets, 0.0, mask=vocab_mask)

        # done = False
        for i in range(max_order):
            start_idx = tl.load(state_start_arcs_ptr + cur_state)
            end_idx = tl.load(state_end_arcs_ptr + cur_state)
            indices = start_idx + vocab_offsets
            mask = indices < end_idx

            cur_ilabels = tl.load(ilabels_ptr + indices, mask=mask)
            cur_weights = tl.load(arcs_weights_ptr + indices, mask=mask)
            cur_to_states = tl.load(to_states_ptr + indices, mask=mask)

            not_final_mask = tl.load(new_states_ptr + batch_i * vocab_size + cur_ilabels, mask=mask, other=0) == -1
            # not_final_mask &= mask
            tl.store(
                scores_ptr + batch_i * vocab_size + cur_ilabels,
                tl.load(scores_ptr + batch_i * vocab_size + cur_ilabels, mask=mask) + cur_weights,
                mask=not_final_mask,
            )
            tl.store(new_states_ptr + batch_i * vocab_size + cur_ilabels, cur_to_states, mask=not_final_mask)

            # done |= (cur_state == start_state)
            # backoff
            cur_backoff_weight = tl.load(backoff_weights_ptr + cur_state)
            not_final_mask = tl.load(new_states_ptr + batch_i * vocab_size + vocab_offsets, mask=vocab_mask, other=0) == -1
            tl.store(
                scores_ptr + batch_i * vocab_size + vocab_offsets,
                tl.load(scores_ptr + batch_i * vocab_size + vocab_offsets, mask=vocab_mask) + cur_backoff_weight,
                mask=not_final_mask,
            )
            cur_state = tl.load(backoff_to_states_ptr + cur_state)


class KenLMWrapper:
    def __init__(self, model_path: Path | str, token_offset=100):
        if not HAVE_KENLM:
            raise Exception(f"No kenlm found, cannot instantiate {self.__class__}")
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
    ) -> Tuple[torch.Tensor, list[list["kenlm.State"]]]:
        batch_size = len(states)
        new_states = [[] for _ in range(len(states))]
        scores = torch.zeros(batch_size, vocab_size)
        for i, state in enumerate(states):
            for label in range(vocab_size):
                score, new_state = self.compute_single_score(state, label)
                scores[i, label] = score
                new_states[i].append(new_state)

        return scores, new_states

    def compute_single_score(self, state: "kenlm.State", label: int) -> Tuple[float, "kenlm.State"]:
        """
        Computes the score for KenLM Ngram language model.
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
        state = self.get_init_state(bos=bos)
        total_score = 0.0
        for label in labels:
            score, state = self.compute_single_score(state=state, label=label)
            total_score += score
        return total_score


class NGram(NamedTuple):
    weight: float
    backoff: float
    symbols: tuple[int, ...]


class Arc(NamedTuple):
    weight: float
    ilabel: int
    to: int


class FastNGramLM(nn.Module):
    def __init__(self, lm_path: Path | str, vocab_size: int, token_offset=100, use_triton=True):
        super().__init__()
        if not use_triton:
            logging.warning("Triton is disabled, falling back to PyTorch. "
                            "NB: version without Triton is not compatible with Cuda graphs, decoding can be slow")
        if not HAVE_TRITON and use_triton:
            logging.warning("Triton not found, falling back to PyTorch")
            use_triton = False
        self.use_triton = use_triton
        self.max_order = 0
        self.token_offset = token_offset
        self.vocab_size = vocab_size

        self.special_symbols = {"<s>": -1, "</s>": -2, "<unk>": -3}

        logging.info(f"FastNGramLM: reading LM {lm_path}")

        special_words_pattern = '|'.join(re.escape(symbol) for symbol in self.special_symbols.keys())
        self._pattern = re.compile(rf'({special_words_pattern}|.)\s?')

        self.ngrams, self.ngram2cnt = self._read_ngrams(lm_path)
        self.max_order = len(self.ngrams)

        self._build_prefix_tree()
        self._prefix_tree_to_torch()

    def _read_ngrams(self, lm_path: Path | str) -> Tuple[list[list[NGram]], dict[int, int]]:
        ngram2cnt_read = defaultdict(int)
        ngram2cnt = defaultdict(int)
        ngrams = []
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

                ngrams[-1].append(self._line_to_ngram(line))
                ngram2cnt_read[cur_order] += 1
            assert ngram2cnt == ngram2cnt_read
            assert len(ngrams) == max_order
            logging.info(f"Loaded model, order={max_order}")
            return ngrams, ngram2cnt

    def _line_to_ngram(self, line: str) -> NGram:
        weight, symbols_str, *backoff_opt = line.split("\t")
        if backoff_opt:
            assert len(backoff_opt) == 1
            backoff = _log_e_score(float(backoff_opt[0]))
        else:
            backoff = 0.0
        weight = _log_e_score(float(weight))
        symbols_re = self._pattern.findall(symbols_str)

        symbols = tuple(
            ord(symbol) - self.token_offset if symbol not in self.special_symbols else self.special_symbols[symbol]
            for symbol in symbols_re
        )
        return NGram(weight=weight, backoff=backoff, symbols=symbols)

    def _build_prefix_tree(self):
        logging.info("FastNGramLM: Building prefix tree")
        self.start_state = 0
        self.bos_state = 1
        self.backoff_id = -10
        self.adjacency: list[dict[int, Arc]] = [dict(), dict()]

        num_states = 2

        states_cache = dict()

        for ngram in self.ngrams[0]:
            assert len(ngram.symbols) == 1
            symbol = ngram.symbols[0]
            if symbol == -1:
                # bos
                self.adjacency[self.start_state][symbol] = Arc(weight=ngram.weight, ilabel=symbol, to=self.bos_state)
                self.adjacency[self.bos_state][self.backoff_id] = Arc(
                    weight=ngram.backoff, ilabel=self.backoff_id, to=self.start_state
                )
                states_cache[ngram.symbols] = self.bos_state
            else:
                assert symbol >= 0 or symbol in {-2, -3}
                to_state = num_states
                num_states += 1
                self.adjacency[self.start_state][symbol] = Arc(weight=ngram.weight, ilabel=symbol, to=to_state)
                self.adjacency.append(
                    {self.backoff_id: Arc(weight=ngram.backoff, ilabel=self.backoff_id, to=self.start_state)}
                )
                states_cache[ngram.symbols] = to_state

        for order in tqdm(range(2, self.max_order + 1)):
            ngram: NGram
            for ngram in self.ngrams[order - 1]:
                state = states_cache[ngram.symbols[:-1]]
                # state = self.start_state
                # for symbol in ngram.symbols[:-1]:
                #     state = self.adjacency[state][symbol].to
                backoff_state = states_cache[ngram.symbols[1:]]
                # backoff_state = self.start_state
                # for symbol in ngram.symbols[1:]:
                #     backoff_state = self.adjacency[backoff_state][symbol].to

                symbol = ngram.symbols[-1]
                if order < self.max_order:
                    to_state = num_states
                    num_states += 1
                    self.adjacency[state][symbol] = Arc(weight=ngram.weight, ilabel=symbol, to=to_state)
                    self.adjacency.append(
                        {self.backoff_id: Arc(weight=ngram.backoff, ilabel=self.backoff_id, to=backoff_state)}
                    )
                    states_cache[ngram.symbols] = to_state
                else:
                    self.adjacency[state][symbol] = Arc(weight=ngram.weight, ilabel=symbol, to=backoff_state)
        self.num_states = num_states

    def _prefix_tree_to_torch(self):
        logging.info("Converting prefix tree to PyTorch")
        num_arcs = sum(
            len(state_arcs) if state != self.start_state else self.vocab_size
            for state, state_arcs in enumerate(self.adjacency)
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

        self.unk_prob = self.adjacency[0][-3].weight

        i = 0
        # TODO: arc to start? +1
        state_order[self.start_state] = 1
        state_order[self.bos_state] = 2
        for ilabel in range(self.vocab_size):
            if ilabel in self.adjacency[self.start_state]:
                arc = self.adjacency[self.start_state][ilabel]
                arcs_weights[i] = arc.weight
                from_states[i] = self.start_state
                to_states[i] = arc.to
                ilabels[i] = arc.ilabel
            else:
                arcs_weights[i] = self.unk_prob
                from_states[i] = self.start_state
                to_states[i] = self.start_state
                ilabels[i] = ilabel
            i += 1
        state_end_arcs[self.start_state] = i

        for state in tqdm(range(0, self.num_states)):
            if state == self.start_state:
                continue
            state_start_arcs[state] = i
            for arc in sorted(self.adjacency[state].values(), key=lambda arc: arc.ilabel):
                # TODO: batch sort in PyTorch?
                if arc.ilabel >= 0:
                    arcs_weights[i] = arc.weight
                    from_states[i] = state
                    to_states[i] = arc.to
                    ilabels[i] = arc.ilabel
                    i += 1
                elif arc.ilabel == -10:
                    # backoff
                    backoff_weights[state] = arc.weight
                    backoff_to_states[state] = arc.to
                    state_order[state] = state_order[arc.to] + 1
                else:
                    continue
            state_end_arcs[state] = i

        self.arcs_weights = nn.Parameter(torch.from_numpy(arcs_weights))
        self.register_buffer("from_states", torch.from_numpy(from_states))
        self.register_buffer("to_states", torch.from_numpy(to_states))
        self.register_buffer("ilabels", torch.from_numpy(ilabels))

        self.backoff_weights = nn.Parameter(torch.from_numpy(backoff_weights))
        self.register_buffer("backoff_to_states", torch.from_numpy(backoff_to_states))

        self.register_buffer("state_start_arcs", torch.from_numpy(state_start_arcs))
        self.register_buffer("state_end_arcs", torch.from_numpy(state_end_arcs))
        self.register_buffer("state_order", torch.from_numpy(state_order))

        assert self.state_order.min().item() == 1
        assert self.state_order.max().item() == self.max_order

    def compute_sentence_score_cpu(self, sentence: list[int], bos=True, verbose=False):
        state = self.bos_state if bos else self.start_state
        weight = 0.0
        for token in sentence:
            if verbose:
                print(f"Token: {token}")
            while token not in self.adjacency[state] and state != self.start_state:
                if verbose:
                    print(f"--State: {state}")
                    print(f"--Backoff: {self.adjacency[state][self.backoff_id].weight}")
                weight += self.adjacency[state][self.backoff_id].weight
                state = self.adjacency[state][self.backoff_id].to
            if state == self.start_state and token not in self.adjacency[state]:
                token = -3  # unk
            if verbose:
                print(f"--Final state: {state}")
                print(f"--add-weight: {self.adjacency[state][token].weight:.4f}")
            weight += self.adjacency[state][token].weight
            state = self.adjacency[state][token].to
        return weight

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
            [batch_size], fill_value=self.bos_state if bos else self.start_state, device=device, dtype=torch.long
        )

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: support gradient?
        with torch.no_grad():
            return self.compute_scores_batch(states=states)

    def compute_scores_batch(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_triton and states.device.type == "cuda":
            return self._compute_scores_batch_triton(states=states)
        if self._custom_kernel is not None and states.device.type == "cuda":
            return self._compute_scores_batch_cuda(states=states)
        return self._compute_scores_batch_pytorch(states=states)

    def _compute_scores_batch_triton(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            start_state=self.start_state,
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

    def _compute_scores_batch_pytorch(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            lm_not_done &= current_states != self.start_state
            out_scores += self.backoff_weights[current_states][:, None] * (~state_found)
            torch.where(lm_not_done, self.backoff_to_states[current_states], current_states, out=current_states)
        return out_scores, out_states


# minimal tests
# TODO: move to testing directory
if __name__ == "__main__":
    arpa_lm_path = "/home/vbataev/code/nemo/check_beam_tdt/tdt_tune1_lmslurp.arpa.tmp.arpa"
    device = torch.device("cuda:0")
    _ = torch.tensor(0, device=device)

    lm = KenLMWrapper(arpa_lm_path)
    gpu_lm = FastNGramLM(arpa_lm_path, vocab_size=1024).to(device)

    with torch.no_grad():
        scores1, states1 = gpu_lm._compute_scores_batch_pytorch(states=gpu_lm.get_init_states(1, bos=True))
        scores2, states2 = gpu_lm._compute_scores_batch_cuda(states=gpu_lm.get_init_states(1, bos=True))
        scores3, states3 = gpu_lm._compute_scores_batch_triton(states=gpu_lm.get_init_states(1, bos=True))
    assert (states1 == states2).all()
    assert torch.allclose(scores1, scores2)
    assert (states1 == states3).all()
    assert torch.allclose(scores1, scores3)

    batch_size = 2
    for _ in tqdm(range(10000)):
        start_state = random.randint(0, gpu_lm.num_states - 1)
        with torch.no_grad():
            scores1, states1 = gpu_lm._compute_scores_batch_pytorch(
                states=torch.full([batch_size], fill_value=start_state, device=device, dtype=torch.int64)
            )
            scores2, states2 = gpu_lm._compute_scores_batch_cuda(
                states=torch.full([batch_size], fill_value=start_state, device=device, dtype=torch.int64)
            )
            scores3, states3 = gpu_lm._compute_scores_batch_triton(
                states=torch.full([batch_size], fill_value=start_state, device=device, dtype=torch.int64)
            )
        assert (states1 == states2).all()
        assert (states1 == states3).all()
        assert torch.allclose(scores1, scores2)
        assert torch.allclose(scores1, scores3)

    for bos in [False, True]:
        for i in range(1024):
            if abs(gpu_lm.compute_sentence_score([i], bos=bos) - lm.compute_sentence_score([i], bos=bos)) > 1e-4:
                print(i, gpu_lm.compute_sentence_score([i], bos=bos), lm.compute_sentence_score([i], bos=bos))

    for sentence in [[25, 70, 12], [58, 41, 186, 293, 306, 999, 163, 264, 689, 683, 999]]:
        for bos in [True, False]:
            score1 = lm.compute_sentence_score(sentence, bos=False)
            score2 = gpu_lm.compute_sentence_score_cpu(sentence, bos=False)
            score3 = gpu_lm.compute_sentence_score(sentence, bos=False)
            assert abs(score1 - score2) < 1e-4
            assert abs(score1 - score3) < 1e-4

        # model_score = gpu_lm.compute_sentence_score(sentence, bos=bos, verbose=True)
        # lm_score = lm.compute_sentence_score(sentence, bos=bos)
        # model_score_batch = 0.0
        # states = gpu_lm.get_init_states(1, bos=bos)
        # for token in sentence:
        #     print(states)
        #     scores, new_states = compute_scores_batch(model, states)
        #     print(scores[0, token])
        #     model_score_batch += scores[0, token]
        #     states[:] = new_states[:, token]
        # print(bos, model_score, lm_score, model_score_batch)
