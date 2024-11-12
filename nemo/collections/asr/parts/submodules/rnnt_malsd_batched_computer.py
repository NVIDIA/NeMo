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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from nemo.collections.asr.parts.ngram_lm import FastNGramLM
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.utils import logging
from nemo.utils.enum import PrettyStrEnum


class BatchedBeamHyps:
    """Class to store batched hypotheses (labels, time_indices, scores) for efficient RNNT decoding"""

    def __init__(
        self,
        batch_size: int,
        init_length: int,
        beam_size: int,
        device: Optional[torch.device] = None,
        float_dtype: Optional[torch.dtype] = None,
    ):
        self._max_length = init_length
        self.beam_size = beam_size

        self.current_lengths = torch.zeros([batch_size, self.beam_size], device=device, dtype=torch.long)
        self.transcript = torch.zeros((batch_size, self.beam_size, self._max_length), device=device, dtype=torch.long)
        self.timesteps = torch.zeros(
            (batch_size, self.beam_size, self._max_length), device=device, dtype=torch.long
        )
        self.scores = torch.zeros([batch_size, self.beam_size], device=device, dtype=float_dtype)

        self.last_timestep = torch.full(
            (batch_size, self.beam_size), -1, device=device, dtype=torch.long
        )
        self.last_timestep_lasts = torch.zeros(
            (batch_size, self.beam_size), device=device, dtype=torch.long
        )
        self._batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.beam_size)
        self._ones_batch = torch.ones_like(self._batch_indices)

    def clear_(self):
        self.current_lengths.fill_(0)
        self.transcript.fill_(0)
        self.timesteps.fill_(0)
        self.scores.fill_(0.0)
        self.last_timestep.fill_(-1)
        self.last_timestep_lasts.fill_(0)

    def _allocate_more(self):
        self.transcript = torch.cat((self.transcript, torch.zeros_like(self.transcript)), dim=-1)
        self.timesteps = torch.cat((self.timesteps, torch.zeros_like(self.timesteps)), dim=-1)
        self._max_length *= 2

    def add_results_masked_(
        self,
        active_mask: torch.Tensor,
        labels: torch.Tensor,
        time_indices: torch.Tensor,
        scores: torch.Tensor,
    ):
        if (self.current_lengths + active_mask).max() >= self._max_length:
            self._allocate_more()
        self.add_results_masked_no_checks_(
            active_mask=active_mask,
            labels=labels,
            time_indices=time_indices,
            scores=scores,
        )

    def add_results_masked_no_checks_(
        self,
        active_mask: torch.Tensor,
        labels: torch.Tensor,
        time_indices: torch.Tensor,
        scores: torch.Tensor,
    ):
        # accumulate scores
        # same as self.scores[active_mask] += scores[active_mask], but non-blocking
        torch.where(active_mask, self.scores + scores, self.scores, out=self.scores)

        # store transcript and timesteps
        self.transcript[self._batch_indices, self.current_lengths] = labels
        self.timesteps[self._batch_indices, self.current_lengths] = time_indices
        # store last observed timestep + number of observation for the current timestep
        # if last_timestep == time_indices, increase; else set to 1
        torch.where(
            torch.logical_and(active_mask, self.last_timestep == time_indices),
            self.last_timestep_lasts + 1,
            self.last_timestep_lasts,
            out=self.last_timestep_lasts,
        )
        torch.where(
            torch.logical_and(active_mask, self.last_timestep != time_indices),
            self._ones_batch,
            self.last_timestep_lasts,
            out=self.last_timestep_lasts,
        )
        # same as: self.last_timestep[active_mask] = time_indices[active_mask], but non-blocking
        torch.where(active_mask, time_indices, self.last_timestep, out=self.last_timestep)
        # increase lengths
        self.current_lengths += active_mask

    def to_hyps_list(self) -> list[rnnt_utils.Hypothesis]:
        num_hyps = self.current_lengths.shape[0]
        best_hyps_ids = torch.argmax(self.scores, dim=1).tolist()
        hypotheses = [
            rnnt_utils.Hypothesis(
                score=self.scores[i, best_hyps_ids[i]].item(),
                # TODO: aggregate hyp
                y_sequence=self.transcript[i, best_hyps_ids[i], : self.current_lengths[i, best_hyps_ids[i]]],
                timestep=[],
                alignments=None,
                dec_state=None,
            )
            for i in range(num_hyps)
        ]
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

    def modified_alsd_beam_torch(self,
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
            init_length=max_time * self.max_symbols if self.max_symbols is not None else max_time,
            device=device,
            float_dtype=float_dtype,
        )
        return batched_hyps.to_hyps_list()


    def _loop_labels_beam_torch(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
    ) -> Tuple[rnnt_utils.BatchedHyps, Optional[rnnt_utils.BatchedAlignments], Any]:
        """
        Pure PyTorch implementation

        Args:
            encoder_output: output from the encoder
            encoder_output_length: lengths of the utterances in `encoder_output`
        """
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
            hyps_beam=self.hyps_beam,
            time_beam=self.time_beam,
            init_length=max_time * self.max_symbols if self.max_symbols is not None else max_time,
            device=device,
            float_dtype=float_dtype,
        )
        # sample state, will be replaced further when the decoding for hypothesis is done
        last_decoder_state = self.decoder.initialize_state(
            encoder_output_projected.unsqueeze(1)
            .expand(-1, self.hyps_beam, -1, -1)
            .reshape(batch_size * self.hyps_beam, max_time, -1)
        )

        # initial state, needed for torch.jit to compile (cannot handle None)
        state = self.decoder.initialize_state(
            encoder_output_projected.unsqueeze(1)
            .expand(-1, self.hyps_beam, -1, -1)
            .reshape(batch_size * self.hyps_beam, max_time, -1)
        )
        # indices of elements in batch (constant)
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)[:, None, None].expand(
            batch_size, self.hyps_beam, self.time_beam
        )
        # last found labels - initially <SOS> (<blank>) symbol
        labels = torch.full([batch_indices, self.hyps_beam], fill_value=self._SOS, device=device, dtype=torch.long)

        # time indices
        time_indices = torch.zeros_like(batch_indices)
        safe_time_indices = torch.zeros_like(time_indices)  # time indices, guaranteed to be < out_len
        time_indices_current_labels = torch.zeros_like(time_indices)
        last_timesteps = (encoder_output_length - 1)[:, None, None].expand_as(batch_indices)

        # masks for utterances in batch
        active_mask: torch.Tensor = (encoder_output_length > 0)[:, None, None].expand_as(batch_indices)

        if self.ngram_lm_batch is not None:
            batch_lm_states = self.ngram_lm_batch.get_init_states(batch_size=batch_size * self.hyps_beam, bos=True)

        # loop while there are active utterances
        while active_mask.any():
            # active_mask_prev.copy_(active_mask, non_blocking=True)
            # stage 1: get decoder (prediction network) output
            decoder_output, state, *_ = self.decoder.predict(
                labels.reshape(-1).unsqueeze(1), state, add_sos=False, batch_size=batch_size
            )
            decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

            if self.ngram_lm_batch is not None:
                lm_scores, batch_lm_states_candidates = self.ngram_lm_batch(
                    states=batch_lm_states
                )  # vocab_size_no_blank
                lm_scores = lm_scores.to(dtype=float_dtype)
                # scores_w_lm, labels_w_lm = (logits[:, :-1] + self.ngram_lm_alpha * lm_scores).max(dim=-1)

            advance_mask = active_mask.clone()
            labels.fill_(value=self._blank_index)
            scores = torch.full(
                [batch_size, self.hyps_beam, self.time_beam], fill_value=0.0, device=device, dtype=float_dtype
            )

            # inner loop: find next non-blank labels (if exist)
            while advance_mask.any():
                # same as: time_indices_current_labels[advance_mask] = time_indices[advance_mask], but non-blocking
                # store current time indices to use further for storing the results
                torch.where(advance_mask, time_indices, time_indices_current_labels, out=time_indices_current_labels)
                logits = (
                    self.joint.joint_after_projection(
                        encoder_output_projected[batch_indices, safe_time_indices].unsqueeze(1),
                        decoder_output,
                    )
                    .squeeze(1)
                    .squeeze(1)
                )
                # get labels (greedy) and scores from current logits, replace labels/scores with new
                # labels[advance_mask] are blank, and we are looking for non-blank labels
                logits = F.log_softmax(logits, dim=-1)
                more_scores, more_labels = logits.max(dim=-1)
                if self.ngram_lm_batch is not None:
                    more_scores_w_lm, more_labels_w_lm = (logits[:, :-1] + self.ngram_lm_alpha * lm_scores).max(dim=-1)
                    torch.where(more_labels == self._blank_index, more_labels, more_labels_w_lm, out=more_labels)
                # same as: labels[advance_mask] = more_labels[advance_mask], but non-blocking
                torch.where(advance_mask, more_labels, labels, out=labels)
                # same as: scores[advance_mask] = more_scores[advance_mask], but non-blocking
                torch.where(advance_mask, more_scores, scores, out=scores)

                blank_mask = labels == self._blank_index
                time_indices += blank_mask
                torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
                torch.less(time_indices, encoder_output_length, out=active_mask)
                torch.logical_and(active_mask, blank_mask, out=advance_mask)

            # stage 3: filter labels and state, store hypotheses
            # select states for hyps that became inactive (is it necessary?)
            # this seems to be redundant, but used in the `loop_frames` output
            # torch.ne(active_mask, active_mask_prev, out=became_inactive_mask)
            # TODO: support last decoder state
            # self.decoder.batch_replace_states_mask(
            #     src_states=state,
            #     dst_states=last_decoder_state,
            #     mask=became_inactive_mask,
            # )

            # store hypotheses
            if self.max_symbols is not None:
                # pre-allocated memory, no need for checks
                batched_hyps.add_results_masked_no_checks_(
                    active_mask,
                    labels,
                    time_indices_current_labels,
                    scores,
                )
            else:
                # auto-adjusted storage
                batched_hyps.add_results_masked_(
                    active_mask,
                    labels,
                    time_indices_current_labels,
                    scores,
                )

            # stage 4: to avoid looping, go to next frame after max_symbols emission
            if self.max_symbols is not None:
                # if labels are non-blank (not end-of-utterance), check that last observed timestep with label:
                # if it is equal to the current time index, and number of observations is >= max_symbols, force blank
                force_blank_mask = torch.logical_and(
                    active_mask,
                    torch.logical_and(
                        torch.logical_and(
                            labels != self._blank_index,
                            batched_hyps.last_timestep_lasts >= self.max_symbols,
                        ),
                        batched_hyps.last_timestep == time_indices,
                    ),
                )
                time_indices += force_blank_mask  # emit blank => advance time indices
                # update safe_time_indices, non-blocking
                torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
                # same as: active_mask = time_indices < encoder_output_length
                torch.less(time_indices, encoder_output_length, out=active_mask)
            if self.ngram_lm_batch is not None:
                torch.where(
                    active_mask,
                    batch_lm_states_candidates[batch_indices, labels * active_mask],
                    batch_lm_states,
                    out=batch_lm_states,
                )
        return batched_hyps, None, last_decoder_state

    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
    ) -> list[rnnt_utils.Hypothesis]:
        return self.modified_alsd_beam_torch(encoder_output=x, encoder_output_length=out_len)
