# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from tqdm import tqdm

from nemo.collections.asr.modules import rnnt_abstract
from nemo.collections.asr.parts.submodules.rnnt_beam_decoding import pack_hypotheses
from nemo.collections.asr.parts.utils.rnnt_utils import (
    Hypothesis,
    NBestHypotheses,
    select_k_expansions,
    is_prefix
)
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import AcousticEncodedRepresentation, HypothesisType, LengthsType, NeuralType
from nemo.utils import logging

try:
    import kenlm

    KENLM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    KENLM_AVAILABLE = False

class BeamTDTInfer(Typing):
    """
    Beam search implementation for Token-andDuration Transducer (TDT) models.

    Sequence level beam decoding or batched-beam decoding, performed auto-repressively
    depending on the search type chosen.

    Args:
        decoder_model: rnnt_utils.AbstractRNNTDecoder implementation.
        joint_model: rnnt_utils.AbstractRNNTJoint implementation.
        durations: list of duration values from TDT model.

        beam_size: number of beams for beam search. Must be a positive integer >= 1.
            If beam size is 1, defaults to stateful greedy search.
            This greedy search might result in slightly different results than
            the greedy results obtained by GreedyRNNTInfer due to implementation differences.

            For accurate greedy results, please use GreedyRNNTInfer or GreedyBatchedRNNTInfer.

        search_type: str representing the type of beam search to perform.
            Must be one of ['beam', 'maes'].

            Algoritm used:

                `beam` - basic beam search strategy. Larger beams generally result in better decoding,
                    however the time required for the search also grows steadily.

                `maes` = modified adaptive expansion search. Please refer to the paper:
                    [Accelerating RNN Transducer Inference via Adaptive Expansion Search](https://ieeexplore.ieee.org/document/9250505)

                    Modified Adaptive Synchronous Decoding (mAES) execution time is adaptive w.r.t the
                    number of expansions (for tokens) required per timestep. The number of expansions can usually
                    be constrained to 1 or 2, and in most cases 2 is sufficient.

                    This beam search technique can possibly obtain superior WER while sacrificing some evaluation time.

        score_norm: bool, whether to normalize the scores of the log probabilities.

        return_best_hypothesis: bool, decides whether to return a single hypothesis (the best out of N),
            or return all N hypothesis (sorted with best score first). The container class changes based
            this flag -
            When set to True (default), returns a single Hypothesis.
            When set to False, returns a NBestHypotheses container, which contains a list of Hypothesis.

        # The following arguments are specific to the chosen `search_type`

        # mAES flags
        maes_num_steps: Number of adaptive steps to take. From the paper, 2 steps is generally sufficient. int > 1.

        maes_prefix_alpha: Maximum prefix length in prefix search. Must be an integer, and is advised to keep this as 1
            in order to reduce expensive beam search cost later. int >= 0.

        maes_expansion_beta: Maximum number of prefix expansions allowed, in addition to the beam size.
            Effectively, the number of hypothesis = beam_size + maes_expansion_beta. Must be an int >= 0,
            and affects the speed of inference since large values will perform large beam search in the next step.

        maes_expansion_gamma: Float pruning threshold used in the prune-by-value step when computing the expansions.
            The default (2.3) is selected from the paper. It performs a comparison (max_log_prob - gamma <= log_prob[v])
            where v is all vocabulary indices in the Vocab set and max_log_prob is the "most" likely token to be
            predicted. Gamma therefore provides a margin of additional tokens which can be potential candidates for
            expansion apart from the "most likely" candidate.
            Lower values will reduce the number of expansions (by increasing pruning-by-value, thereby improving speed
            but hurting accuracy). Higher values will increase the number of expansions (by reducing pruning-by-value,
            thereby reducing speed but potentially improving accuracy). This is a hyper parameter to be experimentally
            tuned on a validation set.

        softmax_temperature: Scales the logits of the joint prior to computing log_softmax.

        preserve_alignments: Bool flag which preserves the history of alignments generated during
            beam decoding (sample). When set to true, the Hypothesis will contain
            the non-null value for `alignments` in it. Here, `alignments` is a List of List of Tensor (of length V + 1).

            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
            U is the number of target tokens for the current timestep Ti.

            NOTE: `preserve_alignments` is an invalid argument for any `search_type`
            other than basic beam search.

        ngram_lm_model: str
            The path to the N-gram LM
        ngram_lm_alpha: float
            Alpha weight of N-gram LM
        tokens_type: str
            Tokenization type ['subword', 'char']
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return {
            "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "partial_hypotheses": [NeuralType(elements_type=HypothesisType(), optional=True)],  # must always be last
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {"predictions": [NeuralType(elements_type=HypothesisType())]}

    def __init__(
        self,
        decoder_model: rnnt_abstract.AbstractRNNTDecoder,
        joint_model: rnnt_abstract.AbstractRNNTJoint,
        durations: list,
        beam_size: int,
        search_type: str = 'default',
        score_norm: bool = True,
        return_best_hypothesis: bool = True,
        maes_num_steps: int = 2,
        maes_prefix_alpha: int = 1,
        maes_expansion_gamma: float = 2.3,
        maes_expansion_beta: int = 2,
        softmax_temperature: float = 1.0,
        preserve_alignments: bool = False,
        ngram_lm_model: Optional[str] = None,
        ngram_lm_alpha: float = 0.0,
    ):
        self.joint = joint_model
        self.decoder = decoder_model
        self.durations = durations

        self.token_offset = 0
        self.search_type = search_type
        self.blank = decoder_model.blank_idx
        self.vocab_size = decoder_model.vocab_size
        self.return_best_hypothesis = return_best_hypothesis

        self.beam_size = beam_size
        self.score_norm = score_norm
        self.max_candidates = beam_size
        self.softmax_temperature = softmax_temperature
        self.preserve_alignments = preserve_alignments
        
        if preserve_alignments:
            raise ValueError("Alignment preservation has not been implemented.")
        if beam_size < 1:
            raise ValueError("Beam search size cannot be less than 1!")

        if self.beam_size == 1:
            logging.info("Beam size of 1 was used, switching to sample level `greedy_search`")
            self.search_algorithm = self.greedy_search
        elif search_type == "default":
            self.search_algorithm = self.default_beam_search
        elif search_type == "tsd":
            raise NotImplementedError("`tsd` (Time Synchronous Decoding) has not been implemented.")
        elif search_type == "alsd":
            raise NotImplementedError("`alsd` (Alignment Length Synchronous Decoding) has not been implemented.")
        elif search_type == "nsc":
            raise NotImplementedError("`nsc` (Constrained Beam Search) has not been implemented.")
        elif search_type == "maes":
            self.search_algorithm = self.modified_adaptive_expansion_search
        else:
            raise NotImplementedError(
                f"The search type ({search_type}) supplied is not supported!\n"
                f"Please use one of : (default, tsd, alsd, nsc)"
            )

        if self.search_type == 'maes':
            self.maes_num_steps = int(maes_num_steps)
            self.maes_prefix_alpha = int(maes_prefix_alpha)
            self.maes_expansion_beta = int(maes_expansion_beta)
            self.maes_expansion_gamma = float(maes_expansion_gamma)
            
            self.max_candidates += maes_expansion_beta
            
            if self.maes_prefix_alpha < 0:
                raise ValueError("`maes_prefix_alpha` must be a positive integer.")

            if self.vocab_size < beam_size + maes_expansion_beta:
                raise ValueError(
                    f"beam_size ({beam_size}) + expansion_beta ({maes_expansion_beta}) "
                    f"should be smaller or equal to vocabulary size ({self.vocab_size})."
                )

            if self.maes_num_steps < 2:
                raise ValueError("`maes_num_steps` must be greater than 1.")

        if ngram_lm_model:
            if search_type != "maes":
                raise ValueError("For decoding with language model `maes` decoding strategy must be chosen.")
            
            if KENLM_AVAILABLE:
                self.ngram_lm = kenlm.Model(ngram_lm_model)
                self.ngram_lm_alpha = ngram_lm_alpha
            else:
                raise ImportError("KenLM package (https://github.com/kpu/kenlm) is not installed. " "Use ngram_lm_model=None.")
        else:
            self.ngram_lm = None

    @typecheck()
    def __call__(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        partial_hypotheses: Optional[List[Hypothesis]] = None,
    ) -> Union[Hypothesis, NBestHypotheses]:
        """Perform general beam search.

        Args:
            encoder_output: Encoded speech features (B, D_enc, T_max)
            encoded_lengths: Lengths of the encoder outputs

        Returns:
            Either a list containing a single Hypothesis (when `return_best_hypothesis=True`,
            otherwise a list containing a single NBestHypotheses, which itself contains a list of
            Hypothesis. This list is sorted such that the best hypothesis is the first element.
        """
        # Preserve decoder and joint training state
        decoder_training_state = self.decoder.training
        joint_training_state = self.joint.training

        with torch.inference_mode():
            # Apply optional preprocessing
            encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)

            self.decoder.eval()
            self.joint.eval()

            hypotheses = []
            with tqdm(
                range(encoder_output.size(0)),
                desc='Beam search progress:',
                total=encoder_output.size(0),
                unit='sample',
                ) as idx_gen:

                _p = next(self.joint.parameters())
                dtype = _p.dtype

                # Decode every sample in the batch independently.
                for batch_idx in idx_gen:
                    inseq = encoder_output[batch_idx : batch_idx + 1, : encoded_lengths[batch_idx], :]  # [1, T, D]
                    logitlen = encoded_lengths[batch_idx]

                    if inseq.dtype != dtype:
                        inseq = inseq.to(dtype=dtype)

                    # Extract partial hypothesis if exists
                    partial_hypothesis = partial_hypotheses[batch_idx] if partial_hypotheses is not None else None

                    # Execute the specific search strategy
                    nbest_hyps = self.search_algorithm(
                        inseq, logitlen, partial_hypotheses=partial_hypothesis
                    )  # sorted list of hypothesis

                    # Prepare the list of hypotheses
                    nbest_hyps = pack_hypotheses(nbest_hyps)

                    # Pack the result
                    if self.return_best_hypothesis:
                        best_hypothesis = nbest_hyps[0]  # type: Hypothesis
                    else:
                        best_hypothesis = NBestHypotheses(nbest_hyps)  # type: NBestHypotheses
                    hypotheses.append(best_hypothesis)

        self.decoder.train(decoder_training_state)
        self.joint.train(joint_training_state)

        return (hypotheses,)

    def greedy_search(
        self, h: torch.Tensor, encoded_lengths: torch.Tensor, partial_hypotheses: Optional[Hypothesis] = None
    ) -> List[Hypothesis]:
        """Greedy search implementation for transducer.
        Generic case when beam size = 1. Results might differ slightly due to implementation details
        as compared to `GreedyRNNTInfer` and `GreedyBatchRNNTInfer`.

        Args:
            h: Encoded speech features (1, T_max, D_enc)

        Returns:
            hyp: 1-best decoding results
        """
        raise NotImplementedError("greedy search has not been implemented")

    def default_beam_search(
        self, encoder_output: torch.Tensor, encoded_lengths: torch.Tensor, partial_hypotheses: Optional[Hypothesis] = None
    ) -> List[Hypothesis]:
        """Beam search implementation for TDT models.

        Args:
            encoder_output: encoder outputs (batch, features, timesteps).
            encoded_lengths: lengths of the outputs from the encoder.
 
        Returns:
            nbest_hyps: N-best decoding results
        """
        debug_mode=True
        if debug_mode:
            print(f"encoded lengths={encoded_lengths}")
        
        # Initialize states
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))
        blank_tensor = torch.tensor([self.blank], device=encoder_output.device, dtype=torch.long)

        # Precompute some constants for blank position
        ids = list(range(self.vocab_size + 1))
        ids.remove(self.blank)

        # Used when blank token is first vs last token
        index_incr = 1 if self.blank == 0 else 0

        # Initialize zero vector states
        dec_state = self.decoder.initialize_state(encoder_output)

        # Initialize first hypothesis for the beam (blank)
        cache = {}
        
        kept_hyps = []
        
        start_hyp = Hypothesis(score=0.0, y_sequence=[self.blank], dec_state=dec_state, timestep=[-1], length=0, last_frame=0)
        kept_hyps.append(start_hyp)
        
        frame_idx = 0
        for frame_idx in range(int(encoded_lengths)):
            hyps = [hyp for hyp in kept_hyps if hyp.last_frame==frame_idx]
            kept_hyps = [hyp for hyp in kept_hyps if hyp.last_frame>frame_idx]
            
            while len(hyps) > 0:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)
                
                if debug_mode:
                    print(f"frame_idx={frame_idx}, score={max_hyp.score}, sequence={max_hyp.y_sequence}, timesteps={max_hyp.timestep}, last_frame={max_hyp.last_frame}")

                # update decoder state and get next score
                hi = encoder_output[:, frame_idx : frame_idx + 1, :]  # [1, 1, D]
                y, state, _ = self.decoder.score_hypothesis(max_hyp, cache)  # [1, 1, D]

                # get next token
                logits = self.joint.joint(hi, y) / self.softmax_temperature
                logp = torch.log_softmax(logits[0, 0, 0, : -len(self.durations)], dim=-1)  # [1, 1, 1, V + 1]
                durations_logp = torch.log_softmax(logits[0, 0, 0, -len(self.durations) :], dim=-1)

                # remove blank token before top k
                top_k = logp[ids].topk(beam_k, dim=-1)

                # Two possible steps - blank token or non-blank token predicted
                logp = (torch.cat((top_k[0], logp[self.blank].unsqueeze(0))), torch.cat((top_k[1] + index_incr, blank_tensor)))
                
                # for each possible step
                for logp, k in zip(*logp):
                    for duration_idx, duration in enumerate(self.durations):
                        if k == self.blank and duration == 0:
                            continue
                        
                        duration_logp = durations_logp[duration_idx]
                        # construct hypothesis for step
                        new_hyp = Hypothesis(
                            score=float(max_hyp.score + logp + duration_logp),
                            y_sequence=max_hyp.y_sequence[:],
                            dec_state=max_hyp.dec_state,
                            lm_state=max_hyp.lm_state,
                            timestep=max_hyp.timestep[:],
                            length=encoded_lengths,
                            last_frame=max_hyp.last_frame)

                        # if current token is blank, don't update sequence, just store the current hypothesis
                        if k == self.blank:
                            new_hyp.last_frame += duration
                            hyps_to_update = kept_hyps
                        elif k != self.blank:
                            new_hyp.dec_state = state
                            new_hyp.y_sequence.append(int(k))
                            new_hyp.timestep.append(frame_idx + duration)
                            new_hyp.last_frame += duration
                            
                            hyps_to_update = hyps if duration == 0 else kept_hyps
                        hyps_to_update.append(new_hyp)
                
                # removing duplicate hypothesis.
                kept_hyps = self.remove_duplicate_hypotheses(kept_hyps)
                
                if (len(hyps) > 0):
                    # keep those hypothesis that have scores greater than next search generation
                    hyps_max = float(max(hyps, key=lambda x: x.score).score)
                    kept_most_prob = sorted([hyp for hyp in kept_hyps if hyp.score > hyps_max], key=lambda x: x.score,)
                    # If enough hypothesis have scores greater than next search generation,
                    # stop beam search.
                    if len(kept_most_prob) >= beam:
                        kept_hyps = kept_most_prob
                        break
       
        if debug_mode:
            print(f"{len(kept_hyps)}, {beam}")
            assert(len(kept_hyps) >= beam )
        return self.sort_nbest(kept_hyps)

    def modified_adaptive_expansion_search(
        self, h: torch.Tensor, encoded_lengths: torch.Tensor, partial_hypotheses: Optional[Hypothesis] = None
    ) -> List[Hypothesis]:
        """
        Based on/modified from https://ieeexplore.ieee.org/document/9250505

        Args:
            h: Encoded speech features (1, T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results
        """
        raise NotImplementedError("maes has not been implemented")
    
    def set_decoding_type(self, decoding_type: str):
        # Please check train_kenlm.py in scripts/asr_language_modeling/ to find out why we need
        # TOKEN_OFFSET for BPE-based models
        if decoding_type == 'subword':
            from nemo.collections.asr.parts.submodules.ctc_beam_decoding import DEFAULT_TOKEN_OFFSET

            self.token_offset = DEFAULT_TOKEN_OFFSET
            
            
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

    def remove_duplicate_hypotheses(self, hyps):
        sorted_hyps = sorted(hyps, key=lambda x: x.score, reverse=True)
        kept_hyps = []
        for hyp in sorted_hyps:
            is_present = False
            for kept_hyp in kept_hyps:
                if kept_hyp.y_sequence == hyp.y_sequence and kept_hyp.last_frame == hyp.last_frame:
                    is_present = True
                    break
            if not is_present:
                kept_hyps.append(hyp)
        return kept_hyps