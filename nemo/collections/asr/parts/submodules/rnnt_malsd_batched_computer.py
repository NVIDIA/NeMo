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
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from nemo.collections.asr.parts.ngram_lm import FastNGramLM
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin
from nemo.utils import logging


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

        self.current_lengths_nb = torch.zeros([batch_size, self.beam_size], device=device, dtype=torch.long)
        self.current_lengths_wb = torch.zeros([batch_size, self.beam_size], device=device, dtype=torch.long)
        self.transcript = torch.zeros((batch_size, self.beam_size, self._max_length), device=device, dtype=torch.long)
        self.transcript_prev_ptr = torch.full(
            (batch_size, self.beam_size, self._max_length), fill_value=-1, device=device, dtype=torch.long
        )
        self.timesteps = torch.zeros((batch_size, self.beam_size, self._max_length), device=device, dtype=torch.long)
        self.scores = torch.zeros([batch_size, self.beam_size], device=device, dtype=float_dtype)

        # self.last_timestep = torch.full((batch_size, self.beam_size), fill_value=-1, device=device, dtype=torch.long)
        self.last_timestep_lasts = torch.zeros((batch_size, self.beam_size), device=device, dtype=torch.long)
        self._batch_indices = (
            torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.beam_size).reshape(-1)
        )
        self._beam_indices = torch.arange(beam_size, device=device).unsqueeze(0).expand(batch_size, -1).reshape(-1)
        self._ones_batch = torch.ones_like(self._batch_indices)

    def clear_(self):
        self.current_lengths_nb.fill_(0)
        self.current_lengths_wb.fill_(0)
        self.transcript.fill_(0)
        self.timesteps.fill_(0)
        self.scores.fill_(0.0)
        self.transcript_prev_ptr.fill_(-1)
        # self.last_timestep.fill_(-1)
        self.last_timestep_lasts.fill_(0)

    def _allocate_more(self):
        self.transcript = torch.cat((self.transcript, torch.zeros_like(self.transcript)), dim=-1)
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
        self.scores.copy_(next_hyps_prob)
        self.transcript[self._batch_indices, self._beam_indices, self.current_lengths_wb.view(-1)] = next_labels
        self.transcript_prev_ptr[self._batch_indices, self._beam_indices, self.current_lengths_wb.view(-1)] = (
            hyps_indices
        )
        self.current_lengths_wb += 1
        extended_with_blank = next_labels == self.blank_index
        extended_with_label = (~extended_with_blank) & (next_labels >= 0)
        self.current_lengths_nb += extended_with_label
        self.last_timestep += extended_with_blank
        self.last_timestep_lasts = torch.where(
            extended_with_blank,
            torch.zeros_like(self.last_timestep_lasts),
            self.last_timestep_lasts + extended_with_label,
        )

    def to_hyps_list(self) -> list[rnnt_utils.Hypothesis]:
        transcript = self.transcript.tolist()
        transcript_prev_ptr = self.transcript_prev_ptr.tolist()
        end_indices = torch.argmax(self.scores, dim=-1).tolist()
        batch_size = self.scores.shape[0]
        hyp_length = self.current_lengths_wb[0, 0].item()
        # TODO: faster parallel aggregation
        hypotheses: list[rnnt_utils.Hypothesis] = []
        for i in range(batch_size):
            transcript = []
            cur_index = end_indices[i]
            for j in range(hyp_length - 1, -1, -1):
                token = transcript[cur_index, j]
                if token > 0 and token != self.blank_index:
                    transcript.append(token)
                cur_index = transcript_prev_ptr[cur_index, j]
            hypotheses.append(
                rnnt_utils.Hypothesis(
                    score=self.scores[i, end_indices[i]].item(),
                    # TODO: aggregate hyp
                    y_sequence=reversed(transcript),
                    timestep=[],
                    alignments=None,
                    dec_state=None,
                )
            )
        return hypotheses


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
    ):
        super().__init__()
        self.decoder = decoder
        self.joint = joint
        self._blank_index = blank_index
        self.beam_size = beam_size
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments
        self.preserve_frame_confidence = preserve_frame_confidence
        self._SOS = self._blank_index
        self._init_confidence_method(confidence_method_cfg=confidence_method_cfg)
        assert self._SOS == self._blank_index  # "blank as pad" algorithm only

        if ngram_lm_model is not None:
            assert self._blank_index == self.joint.num_classes_with_blank - self.joint.num_extra_outputs - 1
            self.ngram_lm_batch = FastNGramLM(lm_path=ngram_lm_model, vocab_size=self._blank_index)
        else:
            self.ngram_lm_batch = None
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
            init_length=max_time * self.max_symbols if self.max_symbols is not None else max_time,
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
        # time_indices_current_labels = torch.zeros_like(time_indices)
        last_timesteps = (encoder_output_length - 1)[:, None].expand_as(batch_indices)
        active_mask = time_indices <= last_timesteps

        if self.ngram_lm_batch is not None:
            batch_lm_states = self.ngram_lm_batch.get_init_states(batch_size=batch_size * self.beam_size, bos=True)
            lm_scores, batch_lm_states_candidates = self.ngram_lm_batch(states=batch_lm_states)  # vocab_size_no_blank
            lm_scores = lm_scores.to(dtype=float_dtype)

        decoder_output, state, *_ = self.decoder.predict(
            last_labels_wb.reshape(-1).unsqueeze(1), None, add_sos=False, batch_size=batch_size * self.beam_size
        )
        decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection
        # decoder_output: [(B x Beam), 1, Dim]

        while active_mask.any():
            logits = (
                self.joint.joint_after_projection(
                    encoder_output_projected[batch_indices.view(-1), safe_time_indices.view(-1)].unsqueeze(1),
                    decoder_output,
                )
                .squeeze(1)
                .squeeze(1)
            )
            log_probs = F.log_softmax(logits, dim=-1)  # [(B x Beam), V]
            if self.ngram_lm_batch:
                log_probs[:, -1] += self.ngram_lm_alpha * lm_scores
            log_probs = log_probs.reshape(batch_size, self.beam_size, -1)
            log_probs_top_k, labels_top_k = torch.topk(log_probs, self.beam_size, dim=-1, largest=True, sorted=True)
            log_probs_blank = log_probs[:, :, self._blank_index]
            # size: batch_size x beam_size x beam_size (k)
            hyps_scores = batched_hyps.scores
            hyps_candidates_prob = hyps_scores.unsqueeze(-1) + log_probs_top_k
            hyps_candidates_prob_forced_blank = hyps_scores + log_probs_blank

            # force add final hyps with the same score to the beam
            hyps_candidates_prob = torch.where(
                active_mask.unsqueeze(-1),
                hyps_candidates_prob,
                torch.full_like(hyps_candidates_prob, fill_value=-float("inf")),
            )
            hyps_candidates_prob[..., 0] = torch.where(
                active_mask,
                hyps_candidates_prob[..., 0],
                hyps_scores,
            )
            labels_top_k = torch.where(
                active_mask.unsqueeze(-1), labels_top_k, torch.full_like(labels_top_k, fill_value=-1)
            )

            # force max_symbols -> extend with blank
            if self.max_symbols is not None:
                force_blank = (batched_hyps.last_timestep_lasts >= self.max_symbols) & active_mask
            else:
                force_blank = torch.full_like(active_mask, fill_value=False)
            # force blank extension with respect to self.max_symbols
            hyps_candidates_prob = torch.where(
                force_blank.unsqueeze(-1),
                torch.full_like(hyps_candidates_prob, fill_value=-float("inf")),
                hyps_candidates_prob,
            )
            hyps_candidates_prob[..., 0] = torch.where(
                force_blank, hyps_candidates_prob_forced_blank, hyps_candidates_prob[..., 0]
            )
            labels_top_k = torch.where(
                force_blank.unsqueeze(-1), torch.full_like(labels_top_k, fill_value=self._blank_index), labels_top_k
            )

            hyps_indices = torch.arange(self.beam_size, dtype=torch.long, device=device)[None, :, None].expand(
                batch_size, -1, self.beam_size
            )

            next_hyps_prob, hyps_candidates_indices = torch.topk(
                hyps_candidates_prob.view(batch_size, -1), k=self.beam_size, largest=True, sorted=True
            )
            hyps_indices = hyps_indices[:, hyps_candidates_indices]
            next_labels = labels_top_k[:, hyps_candidates_indices]

            batched_hyps.add_results_(hyps_indices, next_labels, next_hyps_prob)

            # update decoder + lm state
            prev_decoder_output = decoder_output
            decoder_output, state, *_ = self.decoder.predict(
                last_labels_wb.reshape(-1).unsqueeze(1), state, add_sos=False, batch_size=batch_size * self.beam_size
            )
            decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection
            torch.where((last_labels_wb == self._blank_index).view(-1), prev_decoder_output, decoder_output)
            self.decoder.batch_replace_states_mask(...)
            # TODO: replace state
            # TODO: lm state
            if self.ngram_lm_batch is not None:
                lm_scores, batch_lm_states_candidates = self.ngram_lm_batch(
                    states=batch_lm_states
                )  # vocab_size_no_blank
                lm_scores = lm_scores.to(dtype=float_dtype)

            active_mask = time_indices <= last_timesteps

        return batched_hyps.to_hyps_list()

    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
    ) -> list[rnnt_utils.Hypothesis]:
        return self.modified_alsd_beam_torch(encoder_output=x, encoder_output_length=out_len)
