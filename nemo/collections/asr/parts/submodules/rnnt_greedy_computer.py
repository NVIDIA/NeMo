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

from types import NoneType
from typing import Optional, Union

import torch
import torch.nn as nn

# from nemo.collections.common.parts.rnn import label_collate
from nemo.collections.asr.parts.utils import rnnt_utils


class GreedyBatchedRNNTLoopLabelsComputer(nn.Module):
    def __init__(
        self,
        decoder,
        joint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments=False,
        preserve_frame_confidence=False,
    ):
        super().__init__()
        self.decoder = decoder
        self.joint = joint
        self._blank_index = blank_index
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments
        self.preserve_frame_confidence = preserve_frame_confidence
        if preserve_frame_confidence:
            raise NotImplementedError
        self._SOS = self._blank_index
        self._sample_state_for_type_check = self.decoder.initialize_state(torch.tensor([1]))

    def forward(
        self, x: torch.Tensor, out_len: torch.Tensor,
    ):
        """
        Optimized batched greedy decoding.
        The main idea: search for next labels for the whole batch (evaluating Joint)
        and thus always evaluate prediction network with maximum possible batch size
        """
        # if partial_hypotheses is not None:
        #     raise NotImplementedError("`partial_hypotheses` support is not implemented")

        batch_size, max_time, _unused = x.shape
        device = x.device

        x = self.joint.project_encoder(x)  # do not recalculate joint projection, project only once

        # Initialize empty hypotheses and all necessary tensors
        batched_hyps = rnnt_utils.BatchedHyps(
            batch_size=batch_size, init_length=max_time, device=x.device, float_dtype=x.dtype
        )
        time_indices = torch.zeros([batch_size], dtype=torch.long, device=device)  # always of batch_size
        active_indices = torch.arange(batch_size, dtype=torch.long, device=device)  # initial: all indices
        labels = torch.full([batch_size], fill_value=self._blank_index, dtype=torch.long, device=device)
        # state = None

        # init additional structs for hypotheses: last decoder state, alignments, frame_confidence
        last_decoder_state = [None for _ in range(batch_size)]

        # alignments: Optional[rnnt_utils.BatchedAlignments]
        use_alignments = self.preserve_alignments or self.preserve_frame_confidence
        # if self.preserve_alignments or self.preserve_frame_confidence:
        alignments = rnnt_utils.BatchedAlignments(
            batch_size=batch_size,
            logits_dim=self.joint.num_classes_with_blank,
            init_length=max_time * 2,  # blank for each timestep + text tokens
            device=x.device,
            float_dtype=x.dtype,
            store_alignments=self.preserve_alignments,
            store_frame_confidence=self.preserve_frame_confidence,
        )
        # else:
        #     alignments = None

        # loop while there are active indices
        start = True

        # state: Tuple[torch.Tensor, torch.Tensor] = (torch.zeros(0), torch.zeros(0))
        state = self.decoder.initialize_state(torch.zeros(batch_size, device=device))
        while active_indices.shape[0] > 0:
            current_batch_size = active_indices.shape[0]
            # stage 1: get decoder (prediction network) output
            if start:
                # start of the loop, SOS symbol is passed into prediction network
                decoder_output, state, *_ = self._pred_step(
                    self._SOS, None, batch_size=current_batch_size, add_sos=False
                )
                start = False
            else:
                decoder_output, state, *_ = self._pred_step(
                    labels.unsqueeze(1), state, batch_size=current_batch_size, add_sos=False
                )
            decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

            # stage 2: get joint output, iteratively seeking for non-blank labels
            # blank label in `labels` tensor means "end of hypothesis" (for this index)
            logits = (
                self._joint_step_after_projection(
                    x[active_indices, time_indices[active_indices]].unsqueeze(1),
                    decoder_output,
                    log_normalize=True if self.preserve_frame_confidence else None,
                )
                .squeeze(1)
                .squeeze(1)
            )
            scores, labels = logits.max(-1)

            # search for non-blank labels using joint, advancing time indices for blank labels
            # checking max_symbols is not needed, since we already forced advancing time indices for such cases
            blank_mask = labels == self._blank_index
            if use_alignments:
                alignments.add_results_(
                    active_indices=active_indices,
                    time_indices=time_indices[active_indices],
                    logits=logits if self.preserve_alignments else None,
                    labels=labels if self.preserve_alignments else None,
                    # confidence=torch.tensor(self._get_confidence(logits), device=device)
                    # if self.preserve_frame_confidence
                    # else None,
                )
            # advance_mask is a mask for current batch for searching non-blank labels;
            # each element is True if non-blank symbol is not yet found AND we can increase the time index
            advance_mask = torch.logical_and(blank_mask, (time_indices[active_indices] + 1 < out_len[active_indices]))
            while advance_mask.any():
                advance_indices = active_indices[advance_mask]
                time_indices[advance_indices] += 1
                logits = (
                    self._joint_step_after_projection(
                        x[advance_indices, time_indices[advance_indices]].unsqueeze(1),
                        decoder_output[advance_mask],
                        log_normalize=True if self.preserve_frame_confidence else None,
                    )
                    .squeeze(1)
                    .squeeze(1)
                )
                # get labels (greedy) and scores from current logits, replace labels/scores with new
                # labels[advance_mask] are blank, and we are looking for non-blank labels
                more_scores, more_labels = logits.max(-1)
                labels[advance_mask] = more_labels
                scores[advance_mask] = more_scores
                if use_alignments:
                    alignments.add_results_(
                        active_indices=advance_indices,
                        time_indices=time_indices[advance_indices],
                        logits=logits if self.preserve_alignments else None,
                        labels=more_labels if self.preserve_alignments else None,
                        # confidence=torch.tensor(self._get_confidence(logits), device=device)
                        # if self.preserve_frame_confidence
                        # else None,
                    )
                blank_mask = labels == self._blank_index
                advance_mask = torch.logical_and(
                    blank_mask, (time_indices[active_indices] + 1 < out_len[active_indices])
                )

            # stage 3: filter labels and state, store hypotheses
            # the only case, when there are blank labels in predictions - when we found the end for some utterances
            if blank_mask.any():
                non_blank_mask = ~blank_mask
                labels = labels[non_blank_mask]
                scores = scores[non_blank_mask]

                # select states for hyps that became inactive (is it necessary?)
                # this seems to be redundant, but used in the `loop_frames` output
                # inactive_global_indices = active_indices[blank_mask]
                # inactive_inner_indices = torch.arange(current_batch_size, device=device, dtype=torch.long)[blank_mask]
                # for idx, batch_idx in zip(inactive_global_indices.cpu().numpy(), inactive_inner_indices.cpu().numpy()):
                #     last_decoder_state[idx] = self.decoder.batch_select_state(state, batch_idx)

                # update active indices and state
                active_indices = active_indices[non_blank_mask]
                state = self.decoder.mask_select_states(state, non_blank_mask)
            # store hypotheses
            batched_hyps.add_results_(
                active_indices, labels, time_indices[active_indices].clone(), scores,
            )

            # stage 4: to avoid looping, go to next frame after max_symbols emission
            if self.max_symbols is not None:
                # if labels are non-blank (not end-of-utterance), check that last observed timestep with label:
                # if it is equal to the current time index, and number of observations is >= max_symbols, force blank
                force_blank_mask = torch.logical_and(
                    torch.logical_and(
                        labels != self._blank_index,
                        batched_hyps.last_timestep_lasts[active_indices] >= self.max_symbols,
                    ),
                    batched_hyps.last_timestep[active_indices] == time_indices[active_indices],
                )
                if force_blank_mask.any():
                    # forced blank is not stored in the alignments following the original implementation
                    time_indices[active_indices[force_blank_mask]] += 1  # emit blank => advance time indices
                    # elements with time indices >= out_len become inactive, remove them from batch
                    still_active_mask = time_indices[active_indices] < out_len[active_indices]
                    active_indices = active_indices[still_active_mask]
                    labels = labels[still_active_mask]
                    state = self.decoder.mask_select_states(state, still_active_mask)

        if use_alignments:
            return batched_hyps, alignments
        else:
            return batched_hyps, None
        # hyps = rnnt_utils.batched_hyps_to_hypotheses(batched_hyps, alignments)
        # preserve last decoder state (is it necessary?)
        # for i, last_state in enumerate(last_decoder_state):
        #     assert last_state is not None
        # hyps[i].dec_state = last_state
        # return hyps

    @torch.no_grad()
    def _pred_step(
        self,
        label: Union[torch.Tensor, int],
        # hidden: Optional[tuple[torch.Tensor, torch.Tensor]],
        # hidden: Optional[list[torch.Tensor]],
        hidden: Union[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor], None],
        add_sos: bool = False,
        batch_size: Optional[int] = None,
    ):
        """
        Common prediction step based on the AbstractRNNTDecoder implementation.

        Args:
            label: (int/torch.Tensor): Label or "Start-of-Signal" token.
            hidden: (Optional torch.Tensor): RNN State vector
            add_sos (bool): Whether to add a zero vector at the begging as "start of sentence" token.
            batch_size: Batch size of the output tensor.

        Returns:
            g: (B, U, H) if add_sos is false, else (B, U + 1, H)
            hid: (h, c) where h is the final sequence hidden state and c is
                the final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)
        """
        if isinstance(label, torch.Tensor):
            # label: [batch, 1]
            # if label.dtype != torch.long:
            label_tensor = label.long()

        else:
            # Label is an integer
            # TODO: fix jit comptatibility ???
            # if label == self._SOS:
            assert label == self._SOS
            return self.decoder.predict(None, None, add_sos=add_sos, batch_size=batch_size)
            # label_tensor = label_collate([[label]], device=None)

        # output: [B, 1, K]
        if isinstance(self._sample_state_for_type_check, tuple):
            assert isinstance(hidden, (tuple, NoneType))
            return self.decoder.predict(label_tensor, hidden, add_sos=add_sos, batch_size=batch_size)

        assert isinstance(hidden, (list, NoneType))
        return self.decoder.predict(label_tensor, hidden, add_sos=add_sos, batch_size=batch_size)

    def _joint_step_after_projection(self, enc, pred, log_normalize: Optional[bool] = None) -> torch.Tensor:
        """
        Common joint step based on AbstractRNNTJoint implementation.

        Args:
            enc: Output of the Encoder model after projection. A torch.Tensor of shape [B, 1, H]
            pred: Output of the Decoder model after projection. A torch.Tensor of shape [B, 1, H]
            log_normalize: Whether to log normalize or not. None will log normalize only for CPU.

        Returns:
             logits of shape (B, T=1, U=1, V + 1)
        """
        with torch.no_grad():
            logits = self.joint.joint_after_projection(enc, pred)

            if log_normalize is None:
                if not logits.is_cuda:  # Use log softmax only if on CPU
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)
            else:
                if log_normalize:
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)

        return logits
