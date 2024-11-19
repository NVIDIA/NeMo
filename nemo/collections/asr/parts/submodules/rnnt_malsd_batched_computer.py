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
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from nemo.collections.asr.parts.ngram_lm import FastNGramLM
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin
from nemo.utils import logging
from nemo.utils.enum import PrettyStrEnum

MINUS_INF = -float("inf")

# https://stackoverflow.com/a/77213071
MULTIPLIER = 6364136223846793005
INCREMENT = 1
# INCREMENT = 1442695040888963407
MODULUS = 2**64


def hash_text(prev_hash: torch.Tensor, add_labels: torch.Tensor) -> torch.Tensor:
    return prev_hash * MULTIPLIER + INCREMENT + add_labels


class BatchedBeamHyps:
    """Class to store batched hypotheses (labels, time_indices, scores) for efficient RNNT decoding"""

    def __init__(
        self,
        batch_size: int,
        init_length: int,
        beam_size: int,
        blank_index: int,
        device: Optional[torch.device] = None,
        float_dtype: Optional[torch.dtype] = None,
    ):
        self._max_length = init_length
        self.beam_size = beam_size
        self.blank_index = blank_index
        self.batch_size = batch_size

        self.current_lengths_nb = torch.zeros([batch_size, self.beam_size], device=device, dtype=torch.long)
        self.current_lengths_wb = torch.zeros([batch_size, self.beam_size], device=device, dtype=torch.long)
        self.transcript_wb = torch.zeros(
            (batch_size, self.beam_size, self._max_length), device=device, dtype=torch.long
        )
        self.transcript_wb_prev_ptr = torch.full(
            (batch_size, self.beam_size, self._max_length), fill_value=-1, device=device, dtype=torch.long
        )
        self.transcript_hash = torch.zeros([batch_size, self.beam_size], device=device, dtype=torch.long)
        self.last_label = torch.full([batch_size, self.beam_size], fill_value=-1, device=device, dtype=torch.long)
        self.timesteps = torch.zeros((batch_size, self.beam_size, self._max_length), device=device, dtype=torch.long)
        # TODO: separate lm scores (is this necessary?)
        self.scores = torch.zeros([batch_size, self.beam_size], device=device, dtype=float_dtype)
        self.scores.fill_(MINUS_INF)
        self.scores[:, 0].fill_(0.0)

        self.next_timestep = torch.zeros((batch_size, self.beam_size), device=device, dtype=torch.long)
        self.last_timestep_lasts = torch.zeros((batch_size, self.beam_size), device=device, dtype=torch.long)

    def clear_(self):
        self.current_lengths_nb.fill_(0)
        self.current_lengths_wb.fill_(0)
        self.transcript_wb.fill_(0)
        self.transcript_wb_prev_ptr.fill_(-1)
        self.transcript_hash.fill_(0)
        self.last_label.fill_(-1)
        self.timesteps.fill_(0)
        self.scores.fill_(MINUS_INF)
        self.scores[:, 0].fill_(0.0)
        self.next_timestep.fill_(0)
        self.last_timestep_lasts.fill_(0)

    def _allocate_more(self):
        self.transcript_wb = torch.cat((self.transcript_wb, torch.zeros_like(self.transcript_wb)), dim=-1)
        self.transcript_wb_prev_ptr = torch.cat(
            (self.transcript_wb_prev_ptr, torch.zeros_like(self.transcript_wb_prev_ptr)), dim=-1
        )
        self.timesteps = torch.cat((self.timesteps, torch.zeros_like(self.timesteps)), dim=-1)
        self._max_length *= 2

    def add_results_(
        self,
        hyps_indices,
        next_labels,
        next_hyps_prob,
    ):
        if (self.current_lengths_wb + 1).max() >= self._max_length:
            self._allocate_more()
        self.add_results_no_checks_(
            hyps_indices=hyps_indices,
            next_labels=next_labels,
            next_hyps_prob=next_hyps_prob,
        )

    def add_results_no_checks_(
        self,
        hyps_indices,
        next_labels,
        next_hyps_prob,
    ):
        # TODO: timesteps
        self.scores.copy_(next_hyps_prob)
        self.transcript_wb.scatter_(dim=-1, index=self.current_lengths_wb.unsqueeze(-1), src=next_labels.unsqueeze(-1))
        self.transcript_wb_prev_ptr.scatter_(
            dim=-1, index=self.current_lengths_wb.unsqueeze(-1), src=hyps_indices.unsqueeze(-1)
        )
        # self.transcript.scatter_(dim=-1, index=self.current_lengths_nb.unsqueeze(-1), src=next_labels.unsqueeze(-1))
        # self.transcript_prev_ptr.scatter_(dim=-1, index=self.current_lengths_nb.unsqueeze(-1), src=hyps_indices.unsqueeze(-1))
        self.current_lengths_wb += 1
        extended_with_blank = next_labels == self.blank_index
        extended_with_label = (~extended_with_blank) & (next_labels >= 0)
        self.current_lengths_nb = (
            torch.gather(self.current_lengths_nb, dim=-1, index=hyps_indices) + extended_with_label
        )
        # self.next_timestep = torch.gather(self.next_timestep, dim=-1, index=hyps_indices) + 1 - extended_with_label
        self.next_timestep.copy_(self.current_lengths_wb - self.current_lengths_nb)
        self.last_timestep_lasts = torch.where(
            extended_with_blank,
            torch.zeros_like(self.last_timestep_lasts),
            torch.gather(self.last_timestep_lasts, dim=-1, index=hyps_indices) + extended_with_label,
        )

        # track last label
        torch.where(
            extended_with_label,
            next_labels,
            torch.gather(self.last_label, dim=-1, index=hyps_indices),
            out=self.last_label,
        )

        prev_transcript_hash = torch.gather(self.transcript_hash, dim=-1, index=hyps_indices)
        new_transcript_hash = hash_text(prev_transcript_hash, next_labels)
        torch.where(extended_with_label, new_transcript_hash, prev_transcript_hash, out=self.transcript_hash)

    def self_recombine_hyps_(self):
        if self.beam_size <= 1:
            return
        # TODO: separate lm scores
        hyps_equal = (
            (self.transcript_hash[:, :, None] == self.transcript_hash[:, None, :])
            & (self.last_label[:, :, None] == self.last_label[:, None, :])
            & (self.current_lengths_nb[:, :, None] == self.current_lengths_nb[:, None, :])
        )

        scores_matrix = torch.where(
            hyps_equal,
            self.scores[:, None, :].expand(self.batch_size, self.beam_size, self.beam_size),
            torch.full_like(self.scores, fill_value=MINUS_INF)[:, :, None],
        )
        scores_argmax = scores_matrix.argmax(-1, keepdim=False)
        scores_to_keep = (
            torch.arange(self.beam_size, device=scores_argmax.device, dtype=torch.long)[None, :] == scores_argmax
        )
        new_scores = torch.logsumexp(scores_matrix, dim=-1, keepdim=False)
        torch.where(scores_to_keep, new_scores, torch.full_like(new_scores, fill_value=MINUS_INF), out=self.scores)

    def recombine_prune_hyps(self, hyps_extenstions_probs, last_labels) -> torch.Tensor:
        if self.beam_size <= 1:
            return hyps_extenstions_probs
        device = hyps_extenstions_probs.device
        extended_with_symbol = (last_labels != self.blank_index) & (last_labels >= 0)
        current_lengths_nb = (self.current_lengths_nb.unsqueeze(-1) + extended_with_symbol).view(
            self.batch_size, self.beam_size * self.beam_size
        )
        prev_hash = self.transcript_hash.unsqueeze(-1).expand_as(last_labels)
        transcript_hash = hash_text(prev_hash, last_labels)
        transcript_hash = torch.where(extended_with_symbol, transcript_hash, prev_hash).view(
            self.batch_size, self.beam_size * self.beam_size
        )

        hyps_extenstions_probs = hyps_extenstions_probs.view(self.batch_size, self.beam_size * self.beam_size)
        last_labels = last_labels.view(self.batch_size, self.beam_size * self.beam_size)
        # TODO: separate lm scores?
        hyps_equal = (
            (transcript_hash[:, :, None] == transcript_hash[:, None, :])
            & (last_labels[:, :, None] == last_labels[:, None, :])
            & (current_lengths_nb[:, :, None] == current_lengths_nb[:, None, :])
        )

        scores_matrix = torch.where(
            hyps_equal,
            hyps_extenstions_probs[:, None, :].expand(
                self.batch_size, self.beam_size * self.beam_size, self.beam_size * self.beam_size
            ),
            torch.full_like(hyps_extenstions_probs, fill_value=MINUS_INF)[:, :, None],
        )
        scores_argmax = scores_matrix.argmax(-1, keepdim=False)
        scores_to_keep = (
            torch.arange(self.beam_size * self.beam_size, device=device, dtype=torch.long)[None, :] == scores_argmax
        )
        scores_to_copy = (hyps_equal.sum(-1) == 1) | torch.isinf(hyps_extenstions_probs)
        new_scores = torch.logsumexp(scores_matrix, dim=-1, keepdim=False)
        # assert (~torch.isnan(new_scores)).all()
        scores = torch.where(scores_to_keep, new_scores, torch.full_like(new_scores, fill_value=MINUS_INF))
        scores = torch.where(scores_to_copy, hyps_extenstions_probs, scores)
        return scores.view(self.batch_size, self.beam_size, self.beam_size)

    def to_hyps_list(self, score_norm: bool = True) -> list[rnnt_utils.Hypothesis]:
        transcript = self.transcript_wb.tolist()
        transcript_wb_prev_ptr = self.transcript_wb_prev_ptr.tolist()
        if score_norm:
            end_indices = torch.argmax(self.scores / self.current_lengths_nb.to(self.scores.dtype), dim=-1).tolist()
        else:
            end_indices = torch.argmax(self.scores, dim=-1).tolist()
        scores = self.scores.tolist()
        batch_size = self.scores.shape[0]
        hyp_length = self.current_lengths_wb[0, 0].cpu().item()
        # TODO: faster parallel aggregation
        # TODO: timesteps
        hypotheses: list[rnnt_utils.Hypothesis] = []
        for batch_i in range(batch_size):
            cur_transcript = []
            cur_index = end_indices[batch_i]
            # hyp_length = self.last_timestep[i, cur_index]
            for j in range(hyp_length - 1, -1, -1):
                token = transcript[batch_i][cur_index][j]
                if token > 0 and token != self.blank_index:
                    cur_transcript.append(token)
                cur_index = transcript_wb_prev_ptr[batch_i][cur_index][j]
            hypotheses.append(
                rnnt_utils.Hypothesis(
                    score=scores[batch_i][end_indices[batch_i]],
                    y_sequence=cur_transcript[::-1],
                    timestep=[],
                    alignments=None,
                    dec_state=None,
                )
            )
        return hypotheses


class BlankLMScoreMode(PrettyStrEnum):
    NO_SCORE = "no_score"
    PRESERVE_BLANK = "preserve_blank"
    LM_WEIGHTED = "lm_weighted"
    LM_WEIGHTED_FULL = "lm_weighted_full"
    LM_MAX = "lm_max"
    LM_TOP_MAX = "lm_top_max"


class ModifiedALSDBatchedRNNTComputer(ConfidenceMethodMixin):
    """
    mALSD decoding: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053040
    """

    def __init__(
        self,
        decoder,
        joint,
        blank_index: int,
        beam_size: int,
        max_symbols_per_step: Optional[int] = 10,
        preserve_alignments=False,
        preserve_frame_confidence=False,
        confidence_method_cfg: Optional[DictConfig] = None,
        ngram_lm_model: Optional[str | Path] = None,
        ngram_lm_alpha: float = 0.0,
        blank_lm_score_mode: Optional[str | BlankLMScoreMode] = None,
        allow_recombine_hyps: bool = True,
        score_norm: bool = True,
    ):
        super().__init__()
        self.decoder = decoder
        self.joint = joint
        self._blank_index = blank_index
        self.beam_size = beam_size
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments
        self.preserve_frame_confidence = preserve_frame_confidence
        self.allow_recombine_hyps = allow_recombine_hyps
        self._SOS = self._blank_index
        self._init_confidence_method(confidence_method_cfg=confidence_method_cfg)
        self.score_norm = score_norm
        assert self._SOS == self._blank_index  # "blank as pad" algorithm only

        if ngram_lm_model is not None:
            assert self._blank_index == self.joint.num_classes_with_blank - self.joint.num_extra_outputs - 1
            self.ngram_lm_batch = FastNGramLM(lm_path=ngram_lm_model, vocab_size=self._blank_index)
            if blank_lm_score_mode is None:
                self.blank_lm_score_mode = BlankLMScoreMode.LM_TOP_MAX
            else:
                self.blank_lm_score_mode = BlankLMScoreMode(blank_lm_score_mode)
            if self.allow_recombine_hyps:
                # TODO: implement separate scores and fix
                logging.warning("Hyps recombination is not implemented yet with LM, setting to false")
                self.allow_recombine_hyps = False
        else:
            self.ngram_lm_batch = None
            self.blank_lm_score_mode = None
        self.ngram_lm_alpha = ngram_lm_alpha
        assert not self.preserve_alignments
        assert not self.preserve_frame_confidence

    def modified_alsd_beam_torch(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
    ) -> list[rnnt_utils.Hypothesis]:
        batch_size, max_time, vocab_size = encoder_output.shape
        device = encoder_output.device
        # TODO: better way?
        if self.ngram_lm_batch is not None:
            self.ngram_lm_batch.to(device)

        # do not recalculate joint projection, project only once
        encoder_output_projected = self.joint.project_encoder(encoder_output)
        float_dtype = encoder_output_projected.dtype

        # init output structures: BatchedHyps (for results), BatchedAlignments + last decoder state
        # init empty batched hypotheses
        batched_hyps = BatchedBeamHyps(
            batch_size=batch_size,
            beam_size=self.beam_size,
            blank_index=self._blank_index,
            init_length=max_time * (self.max_symbols + 1) if self.max_symbols is not None else max_time,
            device=device,
            float_dtype=float_dtype,
        )

        last_labels_wb = torch.full(
            [batch_size, self.beam_size], fill_value=self._SOS, device=device, dtype=torch.long
        )

        batch_indices = (
            torch.arange(batch_size, dtype=torch.long, device=device)[:, None]
            .expand(batch_size, self.beam_size)
            .clone()
        )
        time_indices = torch.zeros_like(batch_indices)
        safe_time_indices = torch.zeros_like(time_indices)  # time indices, guaranteed to be < out_len
        last_timesteps = (encoder_output_length - 1)[:, None].expand_as(batch_indices)
        active_mask = time_indices <= last_timesteps

        if self.ngram_lm_batch is not None:
            batch_lm_states = self.ngram_lm_batch.get_init_states(batch_size=batch_size * self.beam_size, bos=True)
            lm_scores, batch_lm_states_candidates = self.ngram_lm_batch(states=batch_lm_states)  # vocab_size_no_blank
            lm_scores = lm_scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1) * self.ngram_lm_alpha

        decoder_output, state, *_ = self.decoder.predict(
            last_labels_wb.reshape(-1).unsqueeze(1), None, add_sos=False, batch_size=batch_size * self.beam_size
        )
        decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection
        # decoder_output: [(B x Beam), 1, Dim]

        step = -1
        while active_mask.any():
            step += 1
            logging.warning(f"Step: {step} {batched_hyps.transcript_wb[:, :, batched_hyps.current_lengths_wb[0, 0].item() - 1]} {batched_hyps.scores}")
            # torch.cuda.set_sync_debug_mode(2)
            # step 1: get joint output + fuse with LM (if present)
            logits = (
                self.joint.joint_after_projection(
                    encoder_output_projected[batch_indices.view(-1), safe_time_indices.view(-1)].unsqueeze(1),
                    decoder_output,
                )
                .squeeze(1)
                .squeeze(1)
            )
            log_probs = F.log_softmax(logits, dim=-1).view(batch_size, self.beam_size, -1)  # [(B x Beam), V]
            if self.ngram_lm_batch is not None:
                if self.blank_lm_score_mode is BlankLMScoreMode.NO_SCORE:
                    log_probs[..., :-1] += lm_scores
                    log_probs_top_k, labels_top_k = torch.topk(
                        log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                    )
                elif self.blank_lm_score_mode is BlankLMScoreMode.PRESERVE_BLANK:
                    _, labels_top_k_no_lm = torch.topk(log_probs, self.beam_size, dim=-1, largest=True, sorted=True)
                    log_probs[..., :-1] += lm_scores
                    _, labels_with_lm_nb_top_k = torch.topk(
                        log_probs[..., :-1], self.beam_size, dim=-1, largest=True, sorted=True
                    )
                    # [(BxBeam), beam]
                    # if blank was in labels_top_k -> add blank (last in beam)
                    labels_top_k = labels_with_lm_nb_top_k
                    labels_top_k[..., -1] = torch.where(
                        (labels_top_k_no_lm == self._blank_index).any(dim=-1),
                        torch.full_like(labels_top_k[..., -1], fill_value=self._blank_index),
                        labels_top_k[..., -1],
                    )
                    log_probs_top_k = torch.gather(log_probs, dim=-1, index=labels_top_k)
                elif self.blank_lm_score_mode is BlankLMScoreMode.LM_WEIGHTED:
                    log_probs[..., :-1] += lm_scores
                    log_probs[..., -1] *= 1 + self.ngram_lm_alpha
                    log_probs_top_k, labels_top_k = torch.topk(
                        log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                    )
                elif self.blank_lm_score_mode is BlankLMScoreMode.LM_WEIGHTED_FULL:
                    blank_logprob = log_probs[..., -1]
                    non_blank_logprob = torch.log1p(-torch.clamp(torch.exp(blank_logprob), max=1.0 - 1e-6))
                    # assert (abs(torch.exp(blank_logprob) + torch.exp(non_blank_logprob) - 1.0) < 1e-5).all()
                    log_probs[..., :-1] += non_blank_logprob.unsqueeze(-1) * self.ngram_lm_alpha + lm_scores
                    log_probs[..., -1] *= 1 + self.ngram_lm_alpha
                    log_probs_top_k, labels_top_k = torch.topk(
                        log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                    )
                elif self.blank_lm_score_mode is BlankLMScoreMode.LM_MAX:
                    log_probs[..., :-1] += lm_scores
                    log_probs[..., -1] += lm_scores.max(dim=-1, keepdim=False).values
                    log_probs_top_k, labels_top_k = torch.topk(
                        log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                    )
                elif self.blank_lm_score_mode is BlankLMScoreMode.LM_TOP_MAX:
                    log_probs[..., :-1] += lm_scores
                    _, labels_with_lm_nb_top_k = torch.topk(
                        log_probs[..., :-1], self.beam_size, dim=-1, largest=True, sorted=True
                    )
                    # [(BxBeam), beam]
                    lm_only_scores = torch.gather(
                        lm_scores,
                        dim=-1,
                        index=labels_with_lm_nb_top_k,
                    )
                    log_probs[..., -1] += lm_only_scores.max(dim=-1, keepdim=False).values
                    log_probs_top_k, labels_top_k = torch.topk(
                        log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                    )
                else:
                    raise NotImplementedError
            else:
                log_probs_top_k, labels_top_k = torch.topk(
                    log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                )

            # step 2: Make hyps candidates. Add new scores to hyps, force blank if necessary, recombine hyps, prune
            # step 2.1: hyps candidates
            log_probs_blank = log_probs[..., self._blank_index]
            # size: batch_size x beam_size x beam_size (k)
            hyps_scores = batched_hyps.scores
            hyps_candidates_prob = hyps_scores.unsqueeze(-1) + log_probs_top_k  # hyps from top-k (top-k-prev x top_k)
            hyps_candidates_prob_forced_blank = (
                hyps_scores + log_probs_blank
            )  # hyps with forced blank (top-k-prev x blank)

            # step 2.2 force add final hyps with the same score to the beam
            # final hyps cannot be extended -> mask with minus inf, copy prev scores; label - set to -1
            hyps_candidates_prob = torch.where(
                active_mask.unsqueeze(-1),
                hyps_candidates_prob,
                torch.full_like(hyps_candidates_prob, fill_value=MINUS_INF),
            )
            hyps_candidates_prob[..., 0] = torch.where(
                active_mask,
                hyps_candidates_prob[..., 0],
                hyps_scores,
            )
            labels_top_k = torch.where(
                active_mask.unsqueeze(-1), labels_top_k, torch.full_like(labels_top_k, fill_value=-1)
            )

            # step 2.3: force blank extension with respect to self.max_symbols
            if self.max_symbols is not None:
                force_blank = (batched_hyps.last_timestep_lasts >= self.max_symbols) & active_mask
            else:
                force_blank = torch.full_like(active_mask, fill_value=False)
            # mask all extensions with -inf
            hyps_candidates_prob = torch.where(
                force_blank.unsqueeze(-1),
                torch.full_like(hyps_candidates_prob, fill_value=MINUS_INF),
                hyps_candidates_prob,
            )
            # first element in beam - score for hyp with forced blank
            hyps_candidates_prob[..., 0] = torch.where(
                force_blank, hyps_candidates_prob_forced_blank, hyps_candidates_prob[..., 0]
            )
            labels_top_k = torch.where(
                force_blank.unsqueeze(-1), torch.full_like(labels_top_k, fill_value=self._blank_index), labels_top_k
            )

            # step 2.4: prune and recombine hyps
            # if self.allow_recombine_hyps:
            #     hyps_candidates_prob = batched_hyps.recombine_prune_hyps(hyps_candidates_prob, labels_top_k)

            # step 2.5: final pruning - get top-k from (top-k x top-k) hyps
            hyps_indices = torch.arange(self.beam_size, dtype=torch.long, device=device)[None, :, None].expand(
                batch_size, -1, self.beam_size
            )
            next_hyps_prob, hyps_candidates_indices = torch.topk(
                hyps_candidates_prob.view(batch_size, -1), k=self.beam_size, largest=True, sorted=True
            )
            hyps_indices = torch.gather(hyps_indices.reshape(batch_size, -1), dim=-1, index=hyps_candidates_indices)
            next_labels = torch.gather(labels_top_k.reshape(batch_size, -1), dim=-1, index=hyps_candidates_indices)

            # step 3: store results
            if self.max_symbols is None:
                batched_hyps.add_results_(hyps_indices, next_labels, next_hyps_prob)
            else:
                batched_hyps.add_results_no_checks_(hyps_indices, next_labels, next_hyps_prob)
            if self.allow_recombine_hyps:
                batched_hyps.self_recombine_hyps_()

            # step 4: update decoder state + decoder output (+ lm state/scores)
            last_labels_wb = torch.where(
                next_labels >= 0, next_labels, torch.full_like(next_labels, fill_value=self._blank_index)
            )
            preserve_state = last_labels_wb == self._blank_index

            # update decoder + lm state
            # decoder_output: [(B x Beam), 1, Dim]
            prev_decoder_output = torch.gather(
                decoder_output.view(batch_size, self.beam_size, 1, -1),
                dim=1,
                index=hyps_indices[:, :, None, None].expand(batch_size, self.beam_size, 1, decoder_output.shape[-1]),
            ).view(batch_size * self.beam_size, 1, -1)
            # TODO: move state aggregation to decoder + support stateless decoder:
            #  self.decoder.batch_aggregate_states_beam(...)
            # state: tuple, each is of [Layers, (BxBeam), Dim]
            prev_state = (
                torch.gather(
                    state[0].view(state[0].shape[0], batch_size, self.beam_size, -1),
                    dim=2,
                    index=hyps_indices[None, :, :, None].expand(
                        state[0].shape[0], batch_size, self.beam_size, state[0].shape[-1]
                    ),
                ).view(state[0].shape[0], batch_size * self.beam_size, -1),
                torch.gather(
                    state[1].view(state[1].shape[0], batch_size, self.beam_size, -1),
                    dim=2,
                    index=hyps_indices[None, :, :, None].expand(
                        state[1].shape[0], batch_size, self.beam_size, state[1].shape[-1]
                    ),
                ).view(state[1].shape[0], batch_size * self.beam_size, -1),
            )

            decoder_output, state, *_ = self.decoder.predict(
                last_labels_wb.reshape(-1).unsqueeze(1),
                prev_state,
                add_sos=False,
                batch_size=batch_size * self.beam_size,
            )
            decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

            decoder_output = torch.where(preserve_state.view(-1)[:, None, None], prev_decoder_output, decoder_output)
            self.decoder.batch_replace_states_mask(
                src_states=prev_state, dst_states=state, mask=preserve_state.view(-1)
            )
            if self.ngram_lm_batch is not None:
                # batch_lm_states: [(BxBeam)]
                # batch_lm_states_candidates: [(BxBeam) x V (without blank)]
                batch_lm_states_candidates = torch.gather(
                    batch_lm_states_candidates.view(batch_size, self.beam_size, -1),
                    dim=1,
                    index=hyps_indices[:, :, None].expand(
                        batch_size, self.beam_size, batch_lm_states_candidates.shape[-1]
                    ),
                )
                batch_lm_states_prev = torch.gather(
                    batch_lm_states.view(batch_size, self.beam_size), dim=1, index=hyps_indices
                )
                last_labels_wb_blank_replaced = torch.where(
                    preserve_state, torch.zeros_like(last_labels_wb), last_labels_wb
                )

                batch_lm_states = torch.gather(
                    batch_lm_states_candidates, dim=-1, index=last_labels_wb_blank_replaced.unsqueeze(-1)
                ).squeeze(-1)
                batch_lm_states = torch.where(preserve_state, batch_lm_states_prev, batch_lm_states).view(-1)

                lm_scores, batch_lm_states_candidates = self.ngram_lm_batch(
                    states=batch_lm_states
                )  # vocab_size_no_blank
                lm_scores = lm_scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1) * self.ngram_lm_alpha

            # step 5: update time indices + active mask
            time_indices = batched_hyps.next_timestep
            torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
            active_mask = time_indices <= last_timesteps
            # torch.cuda.set_sync_debug_mode(0)

        return batched_hyps.to_hyps_list(score_norm=self.score_norm)

    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
    ) -> list[rnnt_utils.Hypothesis]:
        return self.modified_alsd_beam_torch(encoder_output=x, encoder_output_length=out_len)
