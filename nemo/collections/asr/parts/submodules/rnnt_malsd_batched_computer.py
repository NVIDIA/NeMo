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

from nemo.collections.asr.parts.submodules.ngram_lm import FastNGramLM
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin
from nemo.collections.asr.parts.utils.rnnt_batched_beam_utils import BatchedBeamHyps, BlankLMScoreMode, PruningMode
from nemo.utils import logging


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
        pruning_mode: Optional[str | PruningMode] = None,
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
        assert not self.preserve_alignments
        assert not self.preserve_frame_confidence

        if ngram_lm_model is not None:
            assert self._blank_index == self.joint.num_classes_with_blank - self.joint.num_extra_outputs - 1
            self.ngram_lm_batch = FastNGramLM.from_arpa(lm_path=ngram_lm_model, vocab_size=self._blank_index)

            self.pruning_mode = PruningMode.EARLY if pruning_mode is None else PruningMode(pruning_mode)
            self.blank_lm_score_mode = (
                BlankLMScoreMode.LM_WEIGHTED_FULL
                if blank_lm_score_mode is None
                else BlankLMScoreMode(blank_lm_score_mode)
            )
        else:
            self.ngram_lm_batch = None
            self.blank_lm_score_mode = None
        self.ngram_lm_alpha = ngram_lm_alpha

    def modified_alsd_beam_torch(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
    ) -> list[rnnt_utils.Hypothesis]:
        batch_size, max_time, vocab_size = encoder_output.shape
        device = encoder_output.device

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
            self.ngram_lm_batch.to(device)

            batch_lm_states = self.ngram_lm_batch.get_init_states(batch_size=batch_size * self.beam_size, bos=True)
            lm_scores, batch_lm_states_candidates = self.ngram_lm_batch.advance(
                states=batch_lm_states
            )  # vocab_size_no_blank
            lm_scores = lm_scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1) * self.ngram_lm_alpha

        decoder_output, state, *_ = self.decoder.predict(
            last_labels_wb.reshape(-1).unsqueeze(1), None, add_sos=False, batch_size=batch_size * self.beam_size
        )
        decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection
        # decoder_output: [(B x Beam), 1, Dim]

        while active_mask.any():
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
                log_probs_top_k, labels_top_k = self.topk_lm(lm_scores, log_probs)
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
                batched_hyps.INACTIVE_SCORE,
            )
            hyps_candidates_prob[..., 0] = torch.where(
                active_mask,
                hyps_candidates_prob[..., 0],
                hyps_scores,
            )
            labels_top_k = torch.where(active_mask.unsqueeze(-1), labels_top_k, -1)

            # step 2.3: force blank extension with respect to self.max_symbols
            if self.max_symbols is not None:
                force_blank = (batched_hyps.last_timestep_lasts >= self.max_symbols) & active_mask
            else:
                force_blank = torch.full_like(active_mask, fill_value=False)
            # mask all extensions with -inf
            hyps_candidates_prob = torch.where(
                force_blank.unsqueeze(-1), batched_hyps.INACTIVE_SCORE, hyps_candidates_prob
            )
            # first element in beam - score for hyp with forced blank
            hyps_candidates_prob[..., 0] = torch.where(
                force_blank, hyps_candidates_prob_forced_blank, hyps_candidates_prob[..., 0]
            )
            labels_top_k = torch.where(force_blank.unsqueeze(-1), self._blank_index, labels_top_k)

            # step 2.4: final pruning - get top-k from (top-k x top-k) hyps
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
            last_labels_wb = torch.where(next_labels >= 0, next_labels, self._blank_index)
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
                last_labels_wb_blank_replaced = torch.where(preserve_state, 0, last_labels_wb)

                batch_lm_states = torch.gather(
                    batch_lm_states_candidates, dim=-1, index=last_labels_wb_blank_replaced.unsqueeze(-1)
                ).squeeze(-1)
                batch_lm_states = torch.where(preserve_state, batch_lm_states_prev, batch_lm_states).view(-1)

                lm_scores, batch_lm_states_candidates = self.ngram_lm_batch.advance(
                    states=batch_lm_states
                )  # vocab_size_no_blank
                lm_scores = lm_scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1) * self.ngram_lm_alpha

            # step 5: update time indices + active mask
            time_indices = batched_hyps.next_timestep
            torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
            active_mask = time_indices <= last_timesteps
            # torch.cuda.set_sync_debug_mode(0)

        return batched_hyps.to_hyps_list(score_norm=self.score_norm)

    def topk_lm(self, lm_scores, log_probs):
        if self.pruning_mode is PruningMode.LATE:
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
                    self._blank_index,
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
                log_probs[..., :-1] += non_blank_logprob.unsqueeze(-1) * self.ngram_lm_alpha + lm_scores
                log_probs[..., -1] *= 1 + self.ngram_lm_alpha
                log_probs_top_k, labels_top_k = torch.topk(
                    log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                )
            elif self.blank_lm_score_mode is BlankLMScoreMode.LM_WEIGHTED_FULL_FIXED_BLANK:
                blank_logprob = log_probs[..., -1]
                non_blank_logprob = torch.log1p(-torch.clamp(torch.exp(blank_logprob), max=1.0 - 1e-6))
                log_probs[..., :-1] += non_blank_logprob.unsqueeze(-1) + lm_scores  # blank prob - the same
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
                raise NotImplementedError(
                    f"The combination of blank scoring mode '{self.blank_lm_score_mode}' "
                    f"and pruning mode '{self.pruning_mode}' is not implemented."
                )
        elif self.pruning_mode is PruningMode.EARLY:
            if self.blank_lm_score_mode is BlankLMScoreMode.NO_SCORE:
                # log_probs[..., :-1] += lm_scores
                log_probs_top_k, labels_top_k = torch.topk(
                    log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                )
                masked_labels = torch.where(labels_top_k == self._blank_index, 0, labels_top_k)
                log_probs_top_k = torch.where(
                    labels_top_k == self._blank_index,
                    log_probs_top_k,
                    log_probs_top_k + torch.gather(lm_scores, dim=-1, index=masked_labels),
                )
            elif self.blank_lm_score_mode is BlankLMScoreMode.LM_WEIGHTED_FULL:
                # choosing topk from acoustic model
                log_probs_top_k, labels_top_k = log_probs.topk(self.beam_size, dim=-1, largest=True, sorted=True)

                blank_logprob = log_probs[..., -1]
                non_blank_logprob = torch.log1p(-torch.clamp(torch.exp(blank_logprob), max=1.0 - 1e-6))

                masked_labels = torch.where(labels_top_k == self._blank_index, 0, labels_top_k)
                log_probs_top_k = torch.where(
                    labels_top_k == self._blank_index,
                    log_probs_top_k * (1 + self.ngram_lm_alpha),
                    log_probs_top_k
                    + non_blank_logprob.unsqueeze(-1) * self.ngram_lm_alpha
                    + torch.gather(lm_scores, dim=-1, index=masked_labels),
                )
            else:
                raise NotImplementedError(
                    f"The combination of blank scoring mode '{self.blank_lm_score_mode}' "
                    f"and pruning mode '{self.pruning_mode}' is not implemented."
                )
        else:
            raise NotImplementedError(f"Pruning mode {self.pruning_mode} is not implemented.")

        return log_probs_top_k, labels_top_k

    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
    ) -> list[rnnt_utils.Hypothesis]:
        return self.modified_alsd_beam_torch(encoder_output=x, encoder_output_length=out_len)
