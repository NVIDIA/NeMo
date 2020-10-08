# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from nemo.collections.asr.parts import rnnt_utils
from nemo.collections.asr.parts.rnnt_utils import Hypothesis
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import *
from nemo.utils import logging


class BeamRNNTInfer(Typing):
    def __init__(
        self,
        decoder_model: rnnt_utils.AbstractRNNTDecoder,
        joint_model: rnnt_utils.AbstractRNNTJoint,
        beam_size: int,
        search_type: str = 'default',
        score_norm: bool = True,
        tsd_max_symbols_per_step: Optional[int] = 50,
        alsd_max_symmetric_expansion: int = 2,
        nsc_max_timesteps_expansion: int = 1,
        nsc_prefix_alpha: int = 1,
    ):
        self.decoder = decoder_model
        self.joint = joint_model

        self.blank = decoder_model.blank_idx
        # self.hidden_size = decoder_model.hidden_size
        self.vocab_size = decoder_model.vocab_size

        if beam_size < 1:
            raise ValueError("Beam search size cannot be less than 1!")

        self.beam_size = beam_size
        self.score_norm = score_norm

        if self.beam_size == 1:
            self.search_algorithm = self.greedy_search
        elif search_type == "default":
            self.search_algorithm = self.default_beam_search
        elif search_type == "tsd":
            self.search_algorithm = self.time_sync_decoding
        elif search_type == "alsd":
            self.search_algorithm = self.align_length_sync_decoding
        elif search_type == "nsc":
            raise NotImplementedError("`nsc` (Constrained Beam Search) has not been implemented.")
            # self.search_algorithm = self.nsc_beam_search
        else:
            raise NotImplementedError(
                f"The search type ({search_type}) supplied is not supported!\n"
                f"Please use one of : (default, tsd, alsd, nsc)"
            )

        if tsd_max_symbols_per_step is None:
            tsd_max_symbols_per_step = -1

        self.tsd_max_symbols_per_step = tsd_max_symbols_per_step
        self.alsd_max_symmetric_expansion = alsd_max_symmetric_expansion
        self.nsc_max_timesteps_expansion = nsc_max_timesteps_expansion
        self.nsc_prefix_alpha = nsc_prefix_alpha

    @typecheck()
    def __call__(self, encoder_output: torch.Tensor, encoded_lengths: torch.Tensor) -> List[Hypothesis]:
        """Perform beam search.
        Args:
            encoder_output: Encoded speech features (B, T_max, D_enc)
            encoded_lengths: Lengths of the encoder outputs
        Returns:
            nbest_hyps: N-best decoding results
        """
        nbest_hyps = self.search_algorithm(encoder_output, encoded_lengths)
        return nbest_hyps

    def sort_nbest(self, hyps: List[Hypothesis]) -> List[Hypothesis]:
        """Sort hypotheses by score or score given sequence length.
        Args:
            hyps: list of hypotheses
        Return:
            hyps: sorted list of hypotheses
        """
        if self.score_norm:
            return sorted(hyps, key=lambda x: x.score / len(x.y_sequence), reverse=True)
        else:
            return sorted(hyps, key=lambda x: x.score, reverse=True)

    def greedy_search(self, h: torch.Tensor) -> List[Hypothesis]:
        """Greedy search implementation for transformer-transducer.
        Args:
            h: Encoded speech features (1, T_max, D_enc)
        Returns:
            hyp: 1-best decoding results
        """
        dec_state = self.decoder.initialize_state(h)

        hyp = Hypothesis(score=0.0, y_sequence=[self.blank], dec_state=dec_state)
        cache = {}

        y, state, _ = self.decoder.score_hypothesis(hyp, cache)

        for i, hi in enumerate(h[0]):
            ytu = torch.log_softmax(self.joint.joint(hi, y), dim=-1)
            logp, pred = torch.max(ytu, dim=-1)

            if pred != self.blank:
                hyp.y_sequence.append(int(pred))
                hyp.score += float(logp)

                hyp.dec_state = state

                y, state, _ = self.decoder.score_hypothesis(hyp, cache)

        return [hyp]

    def default_beam_search(self, h: torch.Tensor, encoded_lengths: torch.Tensor) -> List[Hypothesis]:
        """Beam search implementation.
        Args:
            x: Encoded speech features (1, T_max, D_enc)
        Returns:
            nbest_hyps: N-best decoding results
        """
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))
        blank_tensor = torch.tensor(self.blank, device=h.device, dtype=torch.long)

        dec_state = self.decoder.initialize_state(h)

        kept_hyps = [Hypothesis(score=0.0, y_sequence=[self.blank], dec_state=dec_state)]
        cache = {}

        for hi in h[0]:
            hyps = kept_hyps
            kept_hyps = []

            while True:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                y, state, lm_tokens = self.decoder.score_hypothesis(max_hyp, cache)

                ytu = torch.log_softmax(self.joint.joint(hi, y), dim=-1)

                top_k = ytu[1:].topk(beam_k, dim=-1)

                ytu = (
                    torch.cat((top_k[0], ytu[0:1])),
                    torch.cat((top_k[1] + 1, blank_tensor)),
                )

                # if self.lm:
                #     lm_state, lm_scores = self.lm.predict(max_hyp.lm_state, lm_tokens)

                for logp, k in zip(*ytu):
                    new_hyp = Hypothesis(
                        score=(max_hyp.score + float(logp)),
                        y_sequence=max_hyp.y_sequence[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                    )

                    if k == self.blank:
                        kept_hyps.append(new_hyp)
                    else:
                        new_hyp.dec_state = state

                        new_hyp.y_sequence.append(int(k))

                        # if self.lm:
                        #     new_hyp.lm_state = lm_state
                        #     new_hyp.score += self.lm_weight * lm_scores[0][k]

                        hyps.append(new_hyp)

                hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted([hyp for hyp in kept_hyps if hyp.score > hyps_max], key=lambda x: x.score,)
                if len(kept_most_prob) >= beam:
                    kept_hyps = kept_most_prob
                    break

        return self.sort_nbest(kept_hyps)

    def time_sync_decoding(self, h: torch.Tensor, encoded_lengths: torch.Tensor) -> List[Hypothesis]:
        """Time synchronous beam search implementation.
        Based on https://ieeexplore.ieee.org/document/9053040
        Args:
            h: Encoded speech features (1, T_max, D_enc)
        Returns:
            nbest_hyps: N-best decoding results
        """
        beam = min(self.beam_size, self.vocab_size)

        beam_state = self.decoder.initialize_state(torch.zeros(beam, device=h.device, dtype=h.dtype))  # [L, B, H]

        B = [Hypothesis(y_sequence=[self.blank], score=0.0, dec_state=beam_state[:, :1, :],)]
        cache = {}

        for hi in h[0]:
            A = []
            C = B

            h_enc = hi.unsqueeze(0)

            for v in range(self.tsd_max_symbols_per_step):
                D = []

                beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score_hypothesis(C, cache)

                beam_logp = torch.log_softmax(self.joint.joint(h_enc, beam_y), dim=-1)
                beam_topk = beam_logp[:, 1:].topk(beam, dim=-1)

                seq_A = [h.y_sequence for h in A]

                for i, hyp in enumerate(C):
                    if hyp.y_sequence not in seq_A:
                        A.append(
                            Hypothesis(
                                score=(hyp.score + float(beam_logp[i, 0])),
                                y_sequence=hyp.y_sequence[:],
                                dec_state=hyp.dec_state,
                                lm_state=hyp.lm_state,
                            )
                        )
                    else:
                        dict_pos = seq_A.index(hyp.y_sequence)

                        A[dict_pos].score = np.logaddexp(A[dict_pos].score, (hyp.score + float(beam_logp[i, 0])))

                if v < self.tsd_max_symbols_per_step:

                    for i, hyp in enumerate(C):
                        for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                            new_hyp = Hypothesis(
                                score=(hyp.score + float(logp)),
                                y_sequence=(hyp.y_sequence + [int(k)]),
                                dec_state=beam_state[:, i : i + 1, :],
                                lm_state=hyp.lm_state,
                            )

                            D.append(new_hyp)

                C = sorted(D, key=lambda x: x.score, reverse=True)[:beam]

            B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]

        return self.sort_nbest(B)

    def align_length_sync_decoding(self, h: torch.Tensor, encoded_lengths: torch.Tensor) -> List[Hypothesis]:
        """Alignment-length synchronous beam search implementation.
        Based on https://ieeexplore.ieee.org/document/9053040
        Args:
            h: Encoded speech features (1, T_max, D_enc)
        Returns:
            nbest_hyps: N-best decoding results
        """
        beam = min(self.beam_size, self.vocab_size)

        h = h[0]
        h_length = int(h.size(0))
        u_max = min(self.alsd_max_symmetric_expansion, (h_length - 1))

        init_tensor = h
        beam_state = self.decoder.init_state(torch.zeros((beam,), device=h.device, dtype=h.dtype))  # [L, B, H]

        B = [Hypothesis(y_sequence=[self.blank], score=0.0, dec_state=beam_state[:, :1, :],)]

        final = []
        cache = {}

        for i in range(h_length + u_max):
            A = []

            B_ = []
            h_states = []
            for hyp in B:
                u = len(hyp.y_sequence) - 1
                t = i - u + 1

                if t > (h_length - 1):
                    continue

                B_.append(hyp)
                h_states.append((t, h[t]))

            if B_:
                beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score_hypothesis(B_, cache)

                h_enc = torch.stack([h[1] for h in h_states])

                beam_logp = torch.log_softmax(self.joint.joint(h_enc, beam_y), dim=-1)
                beam_topk = beam_logp[:, 1:].topk(beam, dim=-1)

                for i, hyp in enumerate(B_):
                    new_hyp = Hypothesis(
                        score=(hyp.score + float(beam_logp[i, 0])),
                        y_sequence=hyp.y_sequence[:],
                        dec_state=hyp.dec_state,
                        lm_state=hyp.lm_state,
                    )

                    A.append(new_hyp)

                    if h_states[i][0] == (h_length - 1):
                        final.append(new_hyp)

                    for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                        new_hyp = Hypothesis(
                            score=(hyp.score + float(logp)),
                            y_sequence=(hyp.y_sequence[:] + [int(k)]),
                            dec_state=self.decoder.select_state(beam_state, i),
                            lm_state=hyp.lm_state,
                        )

                        A.append(new_hyp)

                B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]
                B = self.recombine_hypotheses(B)

        if final:
            return self.sort_nbest(final)
        else:
            return B

    def recombine_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Recombine hypotheses with equivalent output sequence.
        Args:
            hypotheses (list): list of hypotheses
        Returns:
           final (list): list of recombined hypotheses
        """
        final = []

        for hyp in hypotheses:
            seq_final = [f.y_sequence for f in final if f.y_sequence]

            if hyp.y_sequence in seq_final:
                seq_pos = seq_final.index(hyp.y_sequence)

                final[seq_pos].score = np.logaddexp(final[seq_pos].score, hyp.score)
            else:
                final.append(hyp)

        return hypotheses
