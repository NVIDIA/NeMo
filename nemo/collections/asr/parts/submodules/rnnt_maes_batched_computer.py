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

from typing import Any, Optional, Tuple

import torch
from omegaconf import DictConfig
from pathlib import Path

from nemo.collections.asr.parts.ngram_lm import FastNGramLM
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin
from nemo.utils import logging
import torch.nn.functional as F

# https://stackoverflow.com/a/77213071
MULTIPLIER = 6364136223846793005
INCREMENT = 1
MODULUS = 2**64

def hash_text(prev_hash: torch.Tensor, add_labels: torch.Tensor) -> torch.Tensor:
    return (prev_hash * MULTIPLIER + INCREMENT + add_labels) 

class BeamBatchedHyps:
    """Class to store batched hypotheses (labels, time_indices, scores) for efficient RNNT decoding"""

    def __init__(
        self,
        batch_size: int,
        beam_size: int,
        max_timesteps: torch.Tensor,
        init_length: int,
        SOS: int,
        device: Optional[torch.device] = None,
        float_dtype: Optional[torch.dtype] = None,
        score_norm: Optional[bool] = True
    ):
        """

        Args:
            batch_size: batch size for hypotheses
            init_length: initial estimate for the length of hypotheses (if the real length is higher, tensors will be reallocated)
            device: device for storing hypotheses
            float_dtype: float type for scores
        """
        if init_length <= 0:
            raise ValueError(f"init_length must be > 0, got {init_length}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        self._max_length = init_length
        self.score_norm = score_norm
        
        self.SOS = SOS
        self.vocab_size = 1025
        
        self.device = device
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.blank_tensor = torch.tensor(self.SOS)

        self._batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1)
        self._beam_indices = torch.arange(beam_size, device=device).unsqueeze(0)

        self.transcripts = torch.zeros((batch_size, beam_size, self._max_length), device=device, dtype=torch.long)
        self.timesteps = torch.zeros((batch_size, beam_size, self._max_length), device=device, dtype=torch.long)
        self.last_timestep = torch.zeros((batch_size, beam_size), device=device, dtype=torch.long)
        self.last_labels = torch.full((batch_size, beam_size), fill_value=self.SOS, device=device, dtype=torch.long)
        self.scores = torch.full((batch_size, beam_size), device=device, dtype=float_dtype, fill_value=float('-inf'))
        self.trandsucer_scores = torch.zeros((batch_size, beam_size, self.vocab_size), device=device, dtype=torch.long)
        self.lm_scores = torch.zeros((batch_size, beam_size, self.vocab_size), device=device, dtype=torch.long)
        self.scores[self._batch_indices, 0] = 0
        
        self.full_transcripts = torch.zeros((batch_size, beam_size, self._max_length), device=device, dtype=torch.long)
        self.full_current_lengths = torch.zeros((batch_size, beam_size), device=device, dtype=torch.long)
        self.current_lengths = torch.zeros((batch_size, beam_size), device=device, dtype=torch.long)

        self.hashes = torch.zeros((batch_size, beam_size), device=device, dtype=torch.long)
        self.prefix_hashes = torch.zeros((batch_size, beam_size), device=device, dtype=torch.long)
        
        self.label_indicies = torch.arange(self._max_length, device=device).unsqueeze(0).unsqueeze(0).repeat((self.batch_size, self.beam_size, 1))
        
        self.max_timesteps = max_timesteps

    def clear_(self):
        self.full_transcripts.fill_(self.SOS)
        self.full_current_lengths.fill_(0)
        self.transcripts.fill_(self.SOS)
        self.last_labels.fill_(self.SOS)
        self.current_lengths.fill_(0)
        self.last_timestep.fill_(0)
        self.timesteps.fill_(0)
        self.scores.fill_(0.0)
        

    def _allocate_more(self):
        """
        Allocate 2x space for tensors, similar to common C++ std::vector implementations
        to maintain O(1) insertion time complexity
        """
        self.full_transcripts = torch.cat((self.full_transcripts, torch.zeros_like(self.full_transcripts)), dim=2)
        self.transcripts = torch.cat((self.transcripts, torch.zeros_like(self.transcripts)), dim=2)
        self.timesteps = torch.cat((self.timesteps, torch.zeros_like(self.timesteps)), dim=2)
        
        self.label_indicies = torch.arange(2 * self._max_length, device=self.label_indicies.device).unsqueeze(0).unsqueeze(0).repeat((self.batch_size, self.beam_size, 1))

        self._max_length *= 2
        
    def append_labels(self,
                        labels_tensor: torch.Tensor,
                        total_logps: torch.Tensor):
        if labels_tensor.dim() == 3:
            num_expanions = labels_tensor.shape[-1]
        elif labels_tensor.dim() == 2:
            num_expanions = 1
            labels_tensor = labels_tensor.unsqueeze(-1)
        else:
            raise ValueError("Wring number of dimensions for labels tensor.")
    
        if self.full_current_lengths.max() > self._max_length - 2:
            self._allocate_more()
        
        for label_idx in range(num_expanions):
            labels = labels_tensor[:, :, label_idx]
            label_mask = labels >= 0
            non_blank_mask = torch.logical_and(label_mask, labels != self.SOS)
            blank_mask = torch.logical_and(label_mask, labels == self.SOS)
            
            self.full_transcripts[self._batch_indices, self._beam_indices, self.full_current_lengths] = \
                torch.where(label_mask, labels, self.full_transcripts[self._batch_indices, self._beam_indices, self.full_current_lengths])
            
            self.last_labels = torch.where(non_blank_mask, labels, self.last_labels)
            self.timesteps[self._batch_indices, self._beam_indices, self.current_lengths] = \
                torch.where(non_blank_mask, self.last_timestep, self.timesteps[self._batch_indices, self._beam_indices, self.current_lengths] )
            self.transcripts[self._batch_indices, self._beam_indices, self.current_lengths] = \
                torch.where(non_blank_mask, labels, self.transcripts[self._batch_indices, self._beam_indices, self.current_lengths])
            
            self.last_timestep = torch.where(blank_mask, self.last_timestep + 1, self.last_timestep)
            self.current_lengths = torch.where(non_blank_mask, self.current_lengths + 1, self.current_lengths)
            self.full_current_lengths = torch.where(label_mask, self.full_current_lengths + 1, self.full_current_lengths)
        
            self.prefix_hashes[non_blank_mask] = self.hashes[non_blank_mask]
            self.hashes[non_blank_mask] = hash_text(self.hashes, labels)[non_blank_mask]
        
        label_mask = labels_tensor >= 0
        non_blank_mask = torch.logical_and(label_mask, labels_tensor != self.SOS)
        blank_mask = torch.logical_and(label_mask, labels_tensor == self.SOS)
        
        self.scores = torch.where(label_mask.any(-1), total_logps, self.scores)


    # def append_labels(self,
    #                  labels: torch.Tensor,
    #                  total_logps: torch.Tensor):
    #     if labels.dim() == 3:
    #         num_expanions = labels.shape[-1]
    #     elif labels.dim() == 2:
    #         num_expanions = 1
    #     else:
    #         raise ValueError("Wring number of dimensions for labels tensor.")
        
    #     if self.full_current_lengths.max() + num_expanions >= self._max_length - 1:
    #         self._allocate_more()
        
    #     label_mask = labels >= 0
    #     non_blank_mask = torch.logical_and(label_mask, labels != self.SOS)
    #     blank_mask = torch.logical_and(label_mask, labels == self.SOS)
    #     to_update_label_idx = torch.logical_and(self.label_indicies >= self.full_current_lengths.unsqueeze(-1),
    #                                             self.label_indicies < self.full_current_lengths.unsqueeze(-1) + num_expanions)
        
    #     self.full_transcripts[to_update_label_idx] = torch.where(label_mask.flatten(), labels.flatten(), self.full_transcripts[to_update_label_idx])
    #     self.scores = torch.where(non_blank_mask.any(dim=-1), total_logps, self.scores)
        
    #     expansion_idx = torch.arange(num_expanions, device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.beam_size, 1)
    #     non_blank_expansion_label_idx = expansion_idx[blank_mask] - 1
    #     non_blank_expansion_label_idx = torch.maximum(non_blank_expansion_label_idx, torch.tensor(0, device=self.device))
    #     last_labels = torch.gather(labels, dim=-1, index=non_blank_expansion_label_idx.view(self.batch_size, self.beam_size, 1)).squeeze(-1)
    #     self.last_labels = torch.where(non_blank_mask.any(dim=-1), last_labels, self.last_labels)
        
    #     self.timesteps[to_update_label_idx] = torch.where(non_blank_mask.flatten(), self.last_timestep, self.timesteps[to_update_label_idx])
    #     self.transcripts[to_update_label_idx] = torch.where(non_blank_mask.flatten(), labels.flatten(), self.transcripts[to_update_label_idx])
        
    #     self.last_timestep = torch.where(blank_mask.any(dim=-1).flatten(), self.last_timestep + 1, self.last_timestep)
    #     self.current_lengths = torch.where(non_blank_mask.any(dim=-1).flatten(), self.current_lengths + non_blank_mask.sum(dim=-1), self.current_lengths)
    #     self.full_current_lengths = torch.where(label_mask.any(dim=-1).flatten(), self.full_current_lengths + label_mask.sum(dim=-1), self.full_current_lengths)
        
    #     # self.prefix_hashes[non_blank_mask] = self.hashes[non_blank_mask]
    #     # self.hashes[non_blank_mask] = hash_text(self.hashes, labels)[non_blank_mask]
        
    
    def update_beam(self,
                     labels: torch.Tensor,
                     label_logps: torch.Tensor,
                     beam_idx: torch.Tensor):
        beam_idx_unsqueezed = beam_idx.unsqueeze(-1).expand(-1, -1, self._max_length)
        beam_idx_unsqueezed_vocab_size = beam_idx.unsqueeze(-1).expand(-1, -1, self.vocab_size)
        
        self.scores = self.scores.gather(dim=1, index=beam_idx)
        self.last_labels = self.last_labels.gather(dim=1, index=beam_idx)
        self.last_timestep = self.last_timestep.gather(dim=1, index=beam_idx)
        self.current_lengths = self.current_lengths.gather(dim=1, index=beam_idx)
        self.full_current_lengths = self.full_current_lengths.gather(dim=1, index=beam_idx)
                
        self.timesteps = self.timesteps.gather(dim=1, index=beam_idx_unsqueezed)
        self.transcripts = self.transcripts.gather(dim=1, index=beam_idx_unsqueezed)
        self.full_transcripts = self.full_transcripts.gather(dim=1, index=beam_idx_unsqueezed)
        # self.decoder_outputs = self.decoder_outputs.gather(dim=1, index=beam_idx_unsqueezed_vocab_size)
        
        self.prefix_hashes = self.prefix_hashes.gather(dim=1, index=beam_idx)
        self.hashes = self.hashes.gather(dim=1, index=beam_idx)
        
        self.append_labels(labels, label_logps)
        # self.self_recombine_hyps_()
        
    def self_recombine_hyps_(self):
        if self.beam_size <= 1:
            return
        
        hashes = torch.where(self.full_current_lengths == 0, -1, self.hashes)
        prefix_hashes = torch.where(self.current_lengths == 0, -2, self.prefix_hashes)
        hyps_equal = (prefix_hashes[:, :, None] == hashes[:, None, :])
        # # TODO: separate lm scores
        # hyps_equal = (
        #     (self.hashes[:, :, None] == self.hashes[:, None, :])
        #     & (self.last_labels[:, :, None] == self.last_labels[:, None, :])
        #     & (self.current_lengths[:, :, None] == self.current_lengths[:, None, :])
        # )

        scores_matrix = torch.where(
            hyps_equal,
            self.scores[:, None, :].expand(self.batch_size, self.beam_size, self.beam_size),
            torch.full_like(self.scores, fill_value=float('-inf'))[:, :, None],
        )
        scores_argmax = scores_matrix.argmax(-1, keepdim=False)
        scores_to_keep = (
            torch.arange(self.beam_size, device=scores_argmax.device, dtype=torch.long)[None, :] == scores_argmax
        )
        new_scores = torch.logsumexp(scores_matrix, dim=-1, keepdim=False)
        torch.where(scores_to_keep, new_scores, torch.full_like(new_scores, fill_value=float('-inf')), out=self.scores)
        
        return hyps_equal
        
    def recombine_hyps(self, labels, label_logps):
        if self.beam_size <= 1:
            return
        non_blanks_mask = labels != self.SOS
        expansion_number = labels.view(self.batch_size, -1).shape[-1]
        expansion_hashes = hash_text(self.hashes.unsqueeze(-1), labels)
        expansion_hashes = torch.where(non_blanks_mask, expansion_hashes, self.hashes.unsqueeze(-1))
        expansions_equal = expansion_hashes.view(self.batch_size, -1)[:, :, None] == expansion_hashes.view(self.batch_size, -1)[:, None, :]
        expansion_scores = (self.scores.unsqueeze(-1) + label_logps).view(self.batch_size, -1)
        expansion_scores = expansion_scores[:, None, :].expand((self.batch_size, expansion_number, expansion_number))
        expansion_scores = torch.where(expansions_equal, expansion_scores, float('-inf'))
        expansion_scores_argmax = expansion_scores.argmax(-1, keepdim=False)
        scores_to_keep = (
            torch.arange(expansion_number, device=expansion_scores_argmax.device, dtype=torch.long)[None, :] == expansion_scores_argmax
        )
        
        new_scores = torch.logsumexp(expansion_scores, dim=-1, keepdim=False)
        recombined_logps = torch.where(scores_to_keep, new_scores, float('-inf'))

        return recombined_logps.view(self.batch_size, self.beam_size, -1)
    
    def recombine_prefixes(self, label_logps: torch.Tensor, active_mask: torch.Tensor):
        assert label_logps.shape[0] == self.batch_size
        assert label_logps.shape[1] == self.beam_size
        assert label_logps.shape[2] == 1025
        
        # mask hashes if batched hyps are empty
        hashes = torch.where(self.full_current_lengths == 0, -1, self.hashes)
        
        # mask prefix hashes if hypotheses of the beam do not have prefixes (e.g. no non-blank labels were appended)
        prefix_hashes = torch.where(self.current_lengths == 0, -2, self.prefix_hashes)
        
        prefix_equal = hashes[:, None, :] == prefix_hashes[:, :, None]
        prefix_labels = self.last_labels.unsqueeze(1).repeat((1, self.beam_size, 1))
        prefix_scores = self.scores.unsqueeze(1).repeat((1, self.beam_size, 1))
        prefix_label_logps = torch.gather(label_logps, dim=-1, index=prefix_labels)
        
        prefix_label_logps = prefix_scores + prefix_label_logps.transpose(dim0=-1, dim1=-2)
        prefix_label_logps = torch.where(prefix_equal, prefix_label_logps, float('-inf'))
        prefix_label_logps = torch.logsumexp(prefix_label_logps, dim=-1)

        to_update_mask = torch.logical_and(active_mask, self.scores != float('-inf'))
        self.scores = torch.where(to_update_mask, torch.logaddexp(self.scores, prefix_label_logps), self.scores)
        
    
    def recombine_hyps_new(self, labels, label_logps):
        if self.beam_size <= 1:
            return
        
        non_blanks_mask = labels != self.SOS
        expansion_number = labels.view(self.batch_size, -1).shape[-1]
                
        # expansions
        expansion_hashes = hash_text(self.hashes.unsqueeze(-1), labels)
        expansion_hashes = torch.where(non_blanks_mask, expansion_hashes, self.hashes.unsqueeze(-1))
        expansions_equal = expansion_hashes.view(self.batch_size, -1)[:, :, None] == expansion_hashes.view(self.batch_size, -1)[:, None, :]
        expansion_scores = (self.scores.unsqueeze(-1) + label_logps).view(self.batch_size, -1)
        expansion_scores = expansion_scores[:, None, :].expand((self.batch_size, expansion_number, expansion_number))
        
        expansion_scores = torch.where(expansions_equal, expansion_scores, float('-inf'))
        expansion_scores, expansion_scores_argmax = expansion_scores.max(dim=-1)
        scores_to_keep = (
            torch.arange(expansion_number, device=expansion_scores_argmax.device, dtype=torch.long)[None, :] == expansion_scores_argmax
        )
        recombined_logps = torch.where(scores_to_keep, expansion_scores, float('-inf'))
        
        # expansion_scores = torch.where(expansions_equal, expansion_scores, float('-inf'))
        # expansion_scores_argmax = expansion_scores.argmax(-1, keepdim=False)
        # scores_to_keep = (
        #     torch.arange(expansion_number, device=expansion_scores_argmax.device, dtype=torch.long)[None, :] == expansion_scores_argmax
        # )
        
        # new_scores = torch.logsumexp(expansion_scores, dim=-1, keepdim=False)
        # recombined_logps = torch.where(scores_to_keep, new_scores, float('-inf'))

        return recombined_logps.view(self.batch_size, self.beam_size, -1)
    
    def recombine_hyps_exact(self, prev_hashes, labels, label_logps):
        if self.beam_size <= 1:
            return
        
        non_blanks_mask = labels != self.SOS
        expansion_number = labels.view(self.batch_size, -1).shape[-1]
                
        # expansions
        expansion_hashes = hash_text(self.hashes.unsqueeze(-1), labels)
        expansion_hashes = torch.where(non_blanks_mask, expansion_hashes, self.hashes.unsqueeze(-1))
        expansions_equal = expansion_hashes.view(self.batch_size, -1)[:, :, None] == expansion_hashes.view(self.batch_size, -1)[:, None, :]
        expansion_scores = (self.scores.unsqueeze(-1) + label_logps).view(self.batch_size, -1)
        expansion_scores = expansion_scores[:, None, :].expand((self.batch_size, expansion_number, expansion_number))
        
        expansion_scores = torch.where(expansions_equal, expansion_scores, float('-inf'))
        expansion_scores, expansion_scores_argmax = expansion_scores.max(dim=-1)
        scores_to_keep = (
            torch.arange(expansion_number, device=expansion_scores_argmax.device, dtype=torch.long)[None, :] == expansion_scores_argmax
        )
        recombined_logps = torch.where(scores_to_keep, expansion_scores, float('-inf'))
        
        # expansion_scores = torch.where(expansions_equal, expansion_scores, float('-inf'))
        # expansion_scores_argmax = expansion_scores.argmax(-1, keepdim=False)
        # scores_to_keep = (
        #     torch.arange(expansion_number, device=expansion_scores_argmax.device, dtype=torch.long)[None, :] == expansion_scores_argmax
        # )
        
        # new_scores = torch.logsumexp(expansion_scores, dim=-1, keepdim=False)
        # recombined_logps = torch.where(scores_to_keep, new_scores, float('-inf'))

        return recombined_logps.view(self.batch_size, self.beam_size, -1)
    
    def get_best_hyps(self):
        result = []
        for batch_idx in self._batch_indices:
            batch_hyps = []
            for beam_idx in range(self.beam_size):
                hyp = rnnt_utils.Hypothesis(
                        score=self.scores[batch_idx, beam_idx].flatten().item(),
                        y_sequence=self.transcripts[batch_idx, beam_idx, :self.current_lengths[batch_idx, beam_idx]].flatten().tolist(),
                        timestep=self.timesteps[batch_idx, beam_idx, :self.current_lengths[batch_idx, beam_idx]].flatten().tolist()
                    )
                batch_hyps.append(hyp)
                
            if self.score_norm:
                result.append(max(batch_hyps, key = lambda hyp: hyp.score / (len(hyp.y_sequence) + 1)))
                # p = max(batch_hyps, key = lambda hyp: hyp.score / (len(hyp.y_sequence) + 1))
                # print("Final")
                # print("Sequence: ", p.y_sequence)
                # print("Timesteps: ", p.timestep)
                # print("Score: ", p.score)
                # print()
            else:
                result.append(max(batch_hyps, key = lambda hyp: hyp.score))
        return result

    def print(self):
        print("-"*100)
        for batch_idx in self._batch_indices.flatten():
            for beam_idx in self._beam_indices.flatten():
                if self.scores[batch_idx, beam_idx].clone().cpu().numpy() > float('-inf'):
                    print(f"({batch_idx}, {beam_idx}). fulltransc: ", self.full_transcripts[batch_idx, beam_idx, :self.full_current_lengths[batch_idx, beam_idx]].clone().cpu().numpy())
                    print(f"({batch_idx}, {beam_idx}). transcript: ", self.transcripts[batch_idx, beam_idx, :self.current_lengths[batch_idx, beam_idx]].clone().cpu().numpy())
                    print(f"({batch_idx}, {beam_idx}). totalscore: ", self.scores[batch_idx, beam_idx].clone().cpu().numpy())
                    print(f"({batch_idx}, {beam_idx}). timestepss: ", self.timesteps[batch_idx, beam_idx, :self.current_lengths[batch_idx, beam_idx]].clone().cpu().numpy())
                    print(f"({batch_idx}, {beam_idx}). currenhash: ", self.hashes[batch_idx, beam_idx].clone().cpu().numpy())
                    print(f"({batch_idx}, {beam_idx}). previohash: ", self.prefix_hashes[batch_idx, beam_idx].clone().cpu().numpy())
                    print()
            print("-"*10)
            
    def print_sorted(self):
        hypotheses_list = []
        for batch_idx in self._batch_indices.flatten():
            for beam_idx in self._beam_indices.flatten():
                if self.scores[batch_idx, beam_idx].clone().cpu().numpy() > float('-inf'):
                    hypotheses_list.append(
                        rnnt_utils.Hypothesis(
                            score=self.scores[batch_idx, beam_idx].flatten().item(),
                            y_sequence=self.transcripts[batch_idx, beam_idx, :self.current_lengths[batch_idx, beam_idx]].flatten().tolist(),
                            timestep=self.timesteps[batch_idx, beam_idx, :self.current_lengths[batch_idx, beam_idx]].flatten().tolist()
                        )
                    )
                    
        for hyp1 in sorted(hypotheses_list, key = lambda x: x.score, reverse=True):
            print("Sequence: ", hyp1.y_sequence)
            print("Timesteps: ", hyp1.timestep)
            print("Score: ", hyp1.score)
            print()
        
def batched_beam_hyps_to_hypotheses(batched_hyps: BeamBatchedHyps):
    return batched_hyps.get_best_hyps()


class BeamBatchedExpansions:
    """Class to store batched hypotheses (labels, time_indices, scores) for efficient RNNT decoding"""

    def __init__(
        self,
        beam_batched_hyps: BeamBatchedHyps
    ):
        max_expansions_count = 2 * beam_batched_hyps.beam_size
        max_labels_count = 4
        
        self.beam_batched_hyps = beam_batched_hyps
        
        self.num_expansions = torch.full((beam_batched_hyps.batch_size, 1), fill_value=0, device=beam_batched_hyps.device)
        self.label_lengths= torch.full((beam_batched_hyps.batch_size, max_expansions_count), fill_value=0, device=beam_batched_hyps.device)
        self.total_logps = torch.full((beam_batched_hyps.batch_size, max_expansions_count), fill_value=0, device=beam_batched_hyps.device)
        self.beam_indices = torch.full((beam_batched_hyps.batch_size, max_expansions_count), fill_value=-1, device=beam_batched_hyps.device)
        
        self.labels = torch.full(
            (beam_batched_hyps.batch_size, max_expansions_count, max_labels_count),
            fill_value=beam_batched_hyps.SOS,
            device=beam_batched_hyps.device)
        
    def append_expansions(self, expansion_labels, expansion_total_logps, expansion_beam_idx, expansion_batch_idx):
        pass
        
        

    def clear_(self):
        self.full_transcripts.fill_(self.SOS)
        self.full_current_lengths.fill_(0)
        self.transcripts.fill_(self.SOS)
        self.last_labels.fill_(self.SOS)
        self.current_lengths.fill_(0)
        self.last_timestep.fill_(0)
        self.timesteps.fill_(0)
        self.scores.fill_(0.0)
        

    def _allocate_more(self):
        """
        Allocate 2x space for tensors, similar to common C++ std::vector implementations
        to maintain O(1) insertion time complexity
        """
        self.full_transcripts = torch.cat((self.full_transcripts, torch.zeros_like(self.full_transcripts)), dim=2)
        self.transcripts = torch.cat((self.transcripts, torch.zeros_like(self.transcripts)), dim=2)
        self.timesteps = torch.cat((self.timesteps, torch.zeros_like(self.timesteps)), dim=2)
        self._max_length *= 2
        

    def append_labels(self,
                     labels: torch.Tensor,
                     total_logps: torch.Tensor):
        if self.full_current_lengths.max() > self._max_length - 2:
            self._allocate_more()
        
        label_mask = labels >= 0
        non_blank_mask = torch.logical_and(label_mask, labels != self.SOS)
        blank_mask = torch.logical_and(label_mask, labels == self.SOS)
        
        self.scores = torch.where(label_mask, total_logps, self.scores)
        self.full_transcripts[self._batch_indices, self._beam_indices, self.full_current_lengths] = \
            torch.where(label_mask, labels, self.full_transcripts[self._batch_indices, self._beam_indices, self.full_current_lengths])
        
        self.last_labels = torch.where(non_blank_mask, labels, self.last_labels)
        self.timesteps[self._batch_indices, self._beam_indices, self.current_lengths] = \
            torch.where(non_blank_mask, self.last_timestep, self.timesteps[self._batch_indices, self._beam_indices, self.current_lengths] )
        self.transcripts[self._batch_indices, self._beam_indices, self.current_lengths] = \
            torch.where(non_blank_mask, labels, self.transcripts[self._batch_indices, self._beam_indices, self.current_lengths])
        
        self.last_timestep = torch.where(blank_mask, self.last_timestep + 1, self.last_timestep)
        self.current_lengths = torch.where(non_blank_mask, self.current_lengths + 1, self.current_lengths)
        self.full_current_lengths = torch.where(label_mask, self.full_current_lengths + 1, self.full_current_lengths)
        
        self.prefix_hashes[non_blank_mask] = self.hashes[non_blank_mask]
        self.hashes[non_blank_mask] = hash_text(self.hashes, labels)[non_blank_mask]
        
    
    def update_beam(self,
                     labels: torch.Tensor,
                     label_logps: torch.Tensor,
                     beam_idx: torch.Tensor):
        beam_idx_unsqueezed = beam_idx.unsqueeze(-1).expand(-1, -1, self._max_length)
        
        self.scores = self.scores.gather(dim=1, index=beam_idx)
        self.last_labels = self.last_labels.gather(dim=1, index=beam_idx)
        self.last_timestep = self.last_timestep.gather(dim=1, index=beam_idx)
        self.current_lengths = self.current_lengths.gather(dim=1, index=beam_idx)
        self.full_current_lengths = self.full_current_lengths.gather(dim=1, index=beam_idx)
                
        self.timesteps = self.timesteps.gather(dim=1, index=beam_idx_unsqueezed)
        self.transcripts = self.transcripts.gather(dim=1, index=beam_idx_unsqueezed)
        self.full_transcripts = self.full_transcripts.gather(dim=1, index=beam_idx_unsqueezed)
        # self.decoder_outputs = self.decoder_outputs.gather(dim=1, index=beam_idx_unsqueezed_vocab_size)
        
        self.prefix_hashes = self.prefix_hashes.gather(dim=1, index=beam_idx)
        self.hashes = self.hashes.gather(dim=1, index=beam_idx)
        
        self.append_labels(labels, label_logps)
        # self.self_recombine_hyps_()
    
    def recombine_prefixes(self, label_logps: torch.Tensor, active_mask: torch.Tensor):
        assert label_logps.shape[0] == self.batch_size
        assert label_logps.shape[1] == self.beam_size
        assert label_logps.shape[2] == 1025
        
        # mask hashes if batched hyps are empty
        hashes = torch.where(self.full_current_lengths == 0, -1, self.hashes)
        # mask prefix hashes if hypotheses of the beam do not have prefixes (e.g. no non-blank labels were appended)
        prefix_hashes = torch.where(self.current_lengths == 0, -2, self.prefix_hashes)
        
        prefix_equal = hashes[:, None, :] == prefix_hashes[:, :, None]
        prefix_labels = self.last_labels.unsqueeze(1).repeat((1, self.beam_size, 1))
        prefix_scores = self.scores.unsqueeze(1).repeat((1, self.beam_size, 1))
        prefix_label_logps = torch.gather(label_logps, dim=-1, index=prefix_labels)
        
        prefix_label_logps = prefix_scores + prefix_label_logps.transpose(dim0=-1, dim1=-2)
        prefix_label_logps = torch.where(prefix_equal, prefix_label_logps, float('-inf'))
        prefix_label_logps = torch.logsumexp(prefix_label_logps, dim=-1)

        self.scores = torch.where(active_mask, torch.logaddexp(self.scores, prefix_label_logps), self.scores)
        
    
    def recombine_hyps_new(self, labels, label_logps):
        if self.beam_size <= 1:
            return
        
        non_blanks_mask = labels != self.SOS
        expansion_number = labels.view(self.batch_size, -1).shape[-1]
                
        # expansions
        expansion_hashes = hash_text(self.hashes.unsqueeze(-1), labels)
        expansion_hashes = torch.where(non_blanks_mask, expansion_hashes, self.hashes.unsqueeze(-1))
        expansions_equal = expansion_hashes.view(self.batch_size, -1)[:, :, None] == expansion_hashes.view(self.batch_size, -1)[:, None, :]
        expansion_scores = (self.scores.unsqueeze(-1) + label_logps).view(self.batch_size, -1)
        expansion_scores = expansion_scores[:, None, :].expand((self.batch_size, expansion_number, expansion_number))
        
        expansion_scores = torch.where(expansions_equal, expansion_scores, float('-inf'))
        expansion_scores, expansion_scores_argmax = expansion_scores.max(dim=-1)
        scores_to_keep = (
            torch.arange(expansion_number, device=expansion_scores_argmax.device, dtype=torch.long)[None, :] == expansion_scores_argmax
        )
        recombined_logps = torch.where(scores_to_keep, expansion_scores, float('-inf'))
        
        # expansion_scores = torch.where(expansions_equal, expansion_scores, float('-inf'))
        # expansion_scores_argmax = expansion_scores.argmax(-1, keepdim=False)
        # scores_to_keep = (
        #     torch.arange(expansion_number, device=expansion_scores_argmax.device, dtype=torch.long)[None, :] == expansion_scores_argmax
        # )
        
        # new_scores = torch.logsumexp(expansion_scores, dim=-1, keepdim=False)
        # recombined_logps = torch.where(scores_to_keep, new_scores, float('-inf'))

        return recombined_logps.view(self.batch_size, self.beam_size, -1)
    
    def recombine_hyps_exact(self, labels, label_logps):
        if self.beam_size <= 1:
            return
        
        non_blanks_mask = labels != self.SOS
        expansion_number = labels.view(self.batch_size, -1).shape[-1]
                
        # expansions
        expansion_hashes = hash_text(self.hashes.unsqueeze(-1), labels)
        expansion_hashes = torch.where(non_blanks_mask, expansion_hashes, self.hashes.unsqueeze(-1))
        expansions_equal = expansion_hashes.view(self.batch_size, -1)[:, :, None] == expansion_hashes.view(self.batch_size, -1)[:, None, :]
        expansion_scores = (self.scores.unsqueeze(-1) + label_logps).view(self.batch_size, -1)
        expansion_scores = expansion_scores[:, None, :].expand((self.batch_size, expansion_number, expansion_number))
        
        expansion_scores = torch.where(expansions_equal, expansion_scores, float('-inf'))
        expansion_scores, expansion_scores_argmax = expansion_scores.max(dim=-1)
        scores_to_keep = (
            torch.arange(expansion_number, device=expansion_scores_argmax.device, dtype=torch.long)[None, :] == expansion_scores_argmax
        )
        recombined_logps = torch.where(scores_to_keep, expansion_scores, float('-inf'))
        
        # expansion_scores = torch.where(expansions_equal, expansion_scores, float('-inf'))
        # expansion_scores_argmax = expansion_scores.argmax(-1, keepdim=False)
        # scores_to_keep = (
        #     torch.arange(expansion_number, device=expansion_scores_argmax.device, dtype=torch.long)[None, :] == expansion_scores_argmax
        # )
        
        # new_scores = torch.logsumexp(expansion_scores, dim=-1, keepdim=False)
        # recombined_logps = torch.where(scores_to_keep, new_scores, float('-inf'))

        return recombined_logps.view(self.batch_size, self.beam_size, -1)
    
    def get_best_hyps(self):
        result = []
        for batch_idx in self._batch_indices:
            batch_hyps = []
            for beam_idx in range(self.beam_size):
                hyp = rnnt_utils.Hypothesis(
                        score=self.scores[batch_idx, beam_idx].flatten().item(),
                        y_sequence=self.transcripts[batch_idx, beam_idx, :self.current_lengths[batch_idx, beam_idx]].flatten().tolist(),
                        timestep=self.timesteps[batch_idx, beam_idx, :self.current_lengths[batch_idx, beam_idx]].flatten().tolist()
                    )
                batch_hyps.append(hyp)
                
            if self.score_norm:
                result.append(max(batch_hyps, key = lambda hyp: hyp.score / (len(hyp.y_sequence) + 1)))
            else:
                result.append(max(batch_hyps, key = lambda hyp: hyp.score))
        return result

    def print(self):
        print("-"*100)
        for batch_idx in self._batch_indices.flatten():
            for beam_idx in self._beam_indices.flatten():
                print(f"({batch_idx}, {beam_idx}). fulltransc: ", self.full_transcripts[batch_idx, beam_idx, :self.full_current_lengths[batch_idx, beam_idx]].clone().cpu().numpy())
                print(f"({batch_idx}, {beam_idx}). transcript: ", self.transcripts[batch_idx, beam_idx, :self.current_lengths[batch_idx, beam_idx]].clone().cpu().numpy())
                print(f"({batch_idx}, {beam_idx}). totalscore: ", self.scores[batch_idx, beam_idx].clone().cpu().numpy())
                print(f"({batch_idx}, {beam_idx}). timestepss: ", self.timesteps[batch_idx, beam_idx, :self.current_lengths[batch_idx, beam_idx]].clone().cpu().numpy())
                print(f"({batch_idx}, {beam_idx}). currenhash: ", self.hashes[batch_idx, beam_idx].clone().cpu().numpy())
                print(f"({batch_idx}, {beam_idx}). previohash: ", self.prefix_hashes[batch_idx, beam_idx].clone().cpu().numpy())
                print()
            print("-"*10)
        
def batched_beam_hyps_to_hypotheses(batched_hyps: BeamBatchedHyps):
    return batched_hyps.get_best_hyps()


class ModifiedAESBatchedRNNTComputer(ConfidenceMethodMixin):
    """
    Label Looping algorithm implementation: optimized batched greedy decoding. Callable.
    Iterates over labels, on each step finding the next non-blank label
    (evaluating Joint multiple times in inner loop); It uses a minimal possible amount of calls
    to prediction network (with maximum possible batch size),
    which makes it especially useful for scaling the prediction network.
    During decoding all active hypotheses ("texts") have the same lengths.
    """

    INITIAL_MAX_TIME = 375  # initial max time, used to init state for Cuda graphs

    def __init__(
        self,
        decoder,
        joint,
        beam_size: int,
        blank_index: int,
        maes_num_steps: int,
        maes_expansion_gamma: float,
        maes_expansion_beta: float,
        preserve_alignments=False,
        preserve_frame_confidence = False, 
        confidence_method_cfg: Optional[DictConfig] = None,
        include_duration_confidence = False,
        ngram_lm_model: Optional[str | Path] = None,
        ngram_lm_alpha: float = 0.0,
        blank_lm_score_mode: Optional[str | rnnt_utils.BlankLMScoreMode] = None,
        pruning_mode: Optional[str | rnnt_utils.PruningMode] = None,
        allow_recombine_hyps: bool = False,
        score_norm: bool = True,
    ):
        """
        Init method.
        Args:
            decoder: Prediction network from RNN-T
            joint: Joint module from RNN-T
            blank_index: index of blank symbol
            durations: list of TDT durations, e.g., [0, 1, 2, 4, 8]
            max_symbols_per_step: max symbols to emit on each step (to avoid infinite looping)
            preserve_alignments: if alignments are needed
            preserve_frame_confidence: if frame confidence is needed
            include_duration_confidence: if duration confidence is needed to be added to the frame confidence
            confidence_method_cfg: config for the confidence
        """
        super().__init__()
        self.decoder = decoder
        self.joint = joint
        # keep durations on CPU to avoid side effects in multi-gpu environments
        self._blank_index = blank_index
        self.preserve_alignments = preserve_alignments
        self.preserve_frame_confidence = preserve_frame_confidence
        self.include_duration_confidence = include_duration_confidence
        self._SOS = self._blank_index
        self._init_confidence_method(confidence_method_cfg=confidence_method_cfg)
        assert self._SOS == self._blank_index  # "blank as pad" algorithm only
        
        self.vocab_size = len(joint.vocabulary)
        self.vocab_with_blank_size = self.vocab_size + 1
        self.beam_size = min(beam_size, self.vocab_size)
        self.score_norm = score_norm
        
        self.ngram_lm_model = ngram_lm_model
        self.ngram_lm_alpha = ngram_lm_alpha
        self.blank_lm_score_mode = blank_lm_score_mode
        self.pruning_mode = rnnt_utils.PruningMode(pruning_mode)
        self.allow_recombine_hyps = allow_recombine_hyps
        
        self.maes_expansion_gamma = maes_expansion_gamma
        self.maes_expansion_beta = maes_expansion_beta
        self.maes_num_steps = maes_num_steps
        
        self.maes_expansion_delta = 2
        self.maes_expansion_max_hyps = self.beam_size + self.maes_expansion_delta
        
        if ngram_lm_model is not None:
            assert self._blank_index == self.joint.num_classes_with_blank - self.joint.num_extra_outputs - 1
            self.ngram_lm_batch = FastNGramLM(lm_path=ngram_lm_model, vocab_size=self._blank_index)
            if blank_lm_score_mode is None:
                self.blank_lm_score_mode = rnnt_utils.BlankLMScoreMode.NO_SCORE
            else:
                self.blank_lm_score_mode = rnnt_utils.BlankLMScoreMode(blank_lm_score_mode)
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

    def batched_adaptive_expansion_search(
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
        batch_size, max_time, _unused = encoder_output.shape
        device = encoder_output.device
        init_length = max_time * self.maes_num_steps if self.maes_num_steps is not None else max_time
        
        if self.ngram_lm_batch is not None:
            self.ngram_lm_batch.to(device)
        
        encoder_output_projected = self.joint.project_encoder(encoder_output)
        float_dtype = encoder_output_projected.dtype
        
        # init empty batched hypotheses
        batched_hyps = BeamBatchedHyps(
            beam_size=self.beam_size,
            batch_size=batch_size,
            max_timesteps=encoder_output_length-1,
            init_length=init_length,
            device=device,
            float_dtype=float_dtype,
            SOS=self._SOS,
            score_norm=self.score_norm
        )
        
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1).repeat(1, self.beam_size)
        expansion_beam_indices = torch.arange(self.beam_size, device=device).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, self.beam_size+self.maes_expansion_beta)
        zeros_column = torch.zeros((batch_size, self.beam_size, 1), device=batched_hyps.device)
        
        time_indices = torch.zeros_like(batch_indices)
        safe_time_indices = torch.zeros_like(time_indices)
        last_timesteps = (encoder_output_length - 1).unsqueeze(-1).repeat(1, self.beam_size)
        active_mask = time_indices <= last_timesteps
        
        lm_scores = None
        if self.ngram_lm_batch is not None:
            batch_lm_states = self.ngram_lm_batch.get_init_states(batch_size=batch_size * self.beam_size, bos=True)
            lm_scores, batch_lm_states_candidates = self.ngram_lm_batch(states=batch_lm_states)  # vocab_size_no_blank
            lm_scores = lm_scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1) * self.ngram_lm_alpha
            lm_scores = torch.cat((lm_scores, zeros_column), dim=2)
        
        prev_state = None
        labels = batched_hyps.last_labels
        decoder_output, state, *_ = self.decoder.predict(labels.view(-1, 1),
                                                                 None,
                                                                 add_sos=False,
                                                                 batch_size=batch_size * self.beam_size)
        decoder_output = self.joint.project_prednet(decoder_output)
        step=0
        while active_mask.any():
            labels = batched_hyps.last_labels 
            to_update = active_mask.clone()
            safe_time_indices = torch.where(active_mask, time_indices, last_timesteps)
            
            logits = self.joint.joint_after_projection(encoder_output_projected[batch_indices.flatten(), safe_time_indices.flatten()].unsqueeze(1), decoder_output)
            logps = torch.log_softmax(logits, dim=-1).squeeze(1).squeeze(1).view(batch_size, self.beam_size, -1)
            
            if lm_scores == None:
                batched_hyps.recombine_prefixes(logps, active_mask)
            else:
                batched_hyps.recombine_prefixes(logps + lm_scores, active_mask)
                
            
            # print("Step: ", step)
            # batched_hyps.print()
            
            expansion_steps=0
            while to_update.any() and expansion_steps < self.maes_num_steps:
                labels, total_logps, beam_idx = self.get_topk_expansions(batch_size, batched_hyps, batch_indices, expansion_beam_indices, lm_scores, to_update, logps)
                
                batched_hyps.update_beam(labels, total_logps, beam_idx)
                
                labels = torch.where(labels == -1, self._SOS, labels)
                blank_mask = labels == self._SOS
                                
                beam_state_idx = (batch_indices * self.beam_size + beam_idx).flatten()
                prev_state = self.decoder.batch_rearrange_states(state, beam_state_idx)
                prev_decoder_output = torch.index_select(decoder_output, dim=0, index=beam_state_idx)
                
                decoder_output, state, *_ = self.decoder.predict(labels.view(-1, 1),
                                                                 prev_state,
                                                                 add_sos=False,
                                                                 batch_size=batch_size * self.beam_size)
                decoder_output = self.joint.project_prednet(decoder_output)
                
                decoder_output = torch.where(blank_mask.flatten().unsqueeze(-1).unsqueeze(-1), prev_decoder_output, decoder_output)
                state = (
                    torch.where(blank_mask.flatten().unsqueeze(0).unsqueeze(-1), prev_state[0], state[0]),
                    torch.where(blank_mask.flatten().unsqueeze(0).unsqueeze(-1), prev_state[1], state[1]),
                )
                
                if self.ngram_lm_batch is not None:
                    batch_lm_states_candidates = torch.index_select(batch_lm_states_candidates, dim=0, index=beam_state_idx)
                    batch_lm_states_prev = torch.index_select(batch_lm_states, dim=0, index=beam_state_idx)
                    labels_w_blank_replaced = torch.where(blank_mask.flatten(), 0, labels.flatten())

                    batch_lm_states = torch.gather(batch_lm_states_candidates, dim=1, index=labels_w_blank_replaced.unsqueeze(-1)).flatten()
                    batch_lm_states = torch.where(blank_mask.flatten(), batch_lm_states_prev, batch_lm_states).view(-1)

                    lm_scores, batch_lm_states_candidates = self.ngram_lm_batch(
                        states=batch_lm_states
                    )
                    lm_scores = lm_scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1) * self.ngram_lm_alpha
                    lm_scores = torch.cat((lm_scores, zeros_column), dim=2)
                
                logits = self.joint.joint_after_projection(encoder_output_projected[batch_indices.flatten(), safe_time_indices.flatten()].unsqueeze(1), decoder_output)
                logps = torch.log_softmax(logits, dim=-1).squeeze(1).squeeze(1).view(batch_size, self.beam_size, -1)
                
                to_update = torch.logical_and(to_update, labels != self._SOS)
                
                expansion_steps += 1
            time_indices += 1
            active_mask = time_indices <= last_timesteps
            
            step+=1

        return batched_hyps.get_best_hyps()

    def get_topk_expansions(self, batch_size, batched_hyps, batch_indices, expansion_beam_indices, lm_scores, to_update, logps):
        if self.pruning_mode is rnnt_utils.PruningMode.LATE:
            if self.ngram_lm_batch is not None:
                if self.blank_lm_score_mode is rnnt_utils.BlankLMScoreMode.NO_SCORE:
                    logps[..., :-1] += lm_scores
                    label_logps, labels = torch.topk(
                                logps, self.beam_size, dim=-1, largest=True, sorted=True
                            )
                    raise NotImplementedError
            else:
                label_logps, labels = logps.topk(self.beam_size, dim=-1, largest=True, sorted=True)
        else:
            label_logps, labels = logps.topk(self.beam_size + self.maes_expansion_beta, dim=-1, largest=True, sorted=True)
            if self.ngram_lm_batch is not None:
                if self.blank_lm_score_mode is rnnt_utils.BlankLMScoreMode.NO_SCORE:
                    label_logps += torch.gather(lm_scores, dim=-1, index=labels)
                else:
                    raise NotImplementedError
        
        label_logps = torch.where(to_update.unsqueeze(-1), label_logps, float('-inf'))
        total_logps = batched_hyps.recombine_hyps_new(labels, label_logps)
        
        total_logps[total_logps <= total_logps.max(dim=-1, keepdim=True).values - self.maes_expansion_gamma] = float('-inf')
        
        labels = torch.where(to_update.unsqueeze(-1), labels, -1)
        total_logps[..., -1] = torch.where(to_update, total_logps[..., -1], batched_hyps.scores)
                    
        total_logps, idx = total_logps.view(batch_size, -1).topk(self.beam_size, dim=-1, largest=True, sorted=True)
        labels = labels.view(batch_size, -1)[batch_indices, idx]
        beam_idx = expansion_beam_indices.view(batch_size, -1)[batch_indices, idx]
        
        return labels, total_logps, beam_idx

    def batched_adaptive_expansion_search_exact(
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
            batch_size, max_time, _unused = encoder_output.shape
            device = encoder_output.device
            init_length = max_time * self.maes_num_steps if self.maes_num_steps is not None else max_time
            
            if self.ngram_lm_batch is not None:
                self.ngram_lm_batch.to(device)
            
            encoder_output_projected = self.joint.project_encoder(encoder_output)
            float_dtype = encoder_output_projected.dtype
            
            # init empty batched hypotheses
            batched_hyps = BeamBatchedHyps(
                beam_size=self.beam_size,
                batch_size=batch_size,
                max_timesteps=encoder_output_length-1,
                init_length=init_length,
                device=device,
                float_dtype=float_dtype,
                SOS=self._SOS,
                score_norm=self.score_norm
            )
            
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1).repeat(1, self.beam_size)
            expansion_label_indices = torch.arange(self.maes_num_steps, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, self.maes_expansion_max_hyps, 1)
            batch_delta_indices = torch.arange(batch_size, device=device).unsqueeze(-1).repeat(1, self.maes_expansion_max_hyps)
            zeros_column = torch.zeros((batch_size, self.beam_size, 1), device=batched_hyps.device)
            zeros_column_delta = torch.zeros((batch_size, self.maes_expansion_max_hyps, 1), device=batched_hyps.device)
            
            time_indices = torch.zeros_like(batch_indices)
            safe_time_indices = torch.zeros_like(time_indices)
            last_timesteps = (encoder_output_length - 1).unsqueeze(-1).repeat(1, self.beam_size)
            active_mask = time_indices <= last_timesteps
            
            lm_scores = None
            if self.ngram_lm_batch is not None:
                batch_lm_states = self.ngram_lm_batch.get_init_states(batch_size=batch_size * self.beam_size, bos=True)
                lm_scores, batch_lm_states_candidates = self.ngram_lm_batch(states=batch_lm_states)  # vocab_size_no_blank
                lm_scores = lm_scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1) * self.ngram_lm_alpha
                lm_scores = torch.cat((lm_scores, zeros_column), dim=2)
            
            prev_state = None
            labels = batched_hyps.last_labels
            decoder_output, state, *_ = self.decoder.predict(labels.view(-1, 1),
                                                                    None,
                                                                    add_sos=False,
                                                                    batch_size=batch_size * self.beam_size)
            decoder_output = self.joint.project_prednet(decoder_output)
            step=0
            while active_mask.any():
                labels = batched_hyps.last_labels
                total_logps = batched_hyps.scores.unsqueeze(-1)
                to_update = active_mask.clone()
                safe_time_indices = torch.where(active_mask, time_indices, last_timesteps)
                
                logits = self.joint.joint_after_projection(encoder_output_projected[batch_indices.flatten(), safe_time_indices.flatten()].unsqueeze(1), decoder_output)
                logps = torch.log_softmax(logits, dim=-1).squeeze(1).squeeze(1).view(batch_size, self.beam_size, -1)
                
                # print("Step: ", step)
                # print("Before prefix search")
                # batched_hyps.print_sorted()
                
                if self.pruning_mode is rnnt_utils.PruningMode.EARLY:
                    if lm_scores != None:
                        batched_hyps.recombine_prefixes(logps + lm_scores, active_mask)
                    else:
                        batched_hyps.recombine_prefixes(logps, active_mask)
                else:
                    if lm_scores != None:
                        if self.blank_lm_score_mode is rnnt_utils.BlankLMScoreMode.LM_WEIGHTED_FULL:
                            total_logps = logps.clone()
                            blank_logprob = logps[..., -1]
                            non_blank_logprob = torch.log1p(-torch.clamp(torch.exp(blank_logprob), max=1.0 - 1e-6))
                            # assert (abs(torch.exp(blank_logprob) + torch.exp(non_blank_logprob) - 1.0) < 1e-5).all()

                            total_logps[..., :-1] += non_blank_logprob.unsqueeze(-1) * self.ngram_lm_alpha + lm_scores[..., :-1]
                            total_logps[..., -1] *= 1 + self.ngram_lm_alpha
                            batched_hyps.recombine_prefixes(total_logps, active_mask)    
                        else:
                            batched_hyps.recombine_prefixes(logps + lm_scores, active_mask)
                    else:
                        batched_hyps.recombine_prefixes(logps, active_mask)
                
                # print("After prefix search")
                # batched_hyps.print_sorted()
                
                expansion_logps = F.pad(batched_hyps.scores, (0, self.maes_expansion_delta, 0, 0), mode='constant', value=float('-inf'))
                expansion_hashes = F.pad(batched_hyps.hashes.unsqueeze(-1), (0, 0, 0, self.maes_expansion_delta, 0, 0), mode='constant', value=-1)
                init_expansion_hashes = expansion_hashes.clone()
                logps = F.pad(logps, (0, 0, 0, self.maes_expansion_delta, 0, 0), mode='constant', value=float('-inf'))
                to_update = F.pad(to_update, (0, self.maes_expansion_delta, 0, 0), mode='constant', value=True)
                safe_time_indices = F.pad(safe_time_indices, (0, self.maes_expansion_delta), mode='replicate')
                
                decoder_output = F.pad(decoder_output.view(batch_size, self.beam_size, decoder_output.shape[1], decoder_output.shape[-1]), (0, 0, 0, 0, 0, self.maes_expansion_delta), mode='constant').view(-1, decoder_output.shape[1], decoder_output.shape[-1])
                state = (
                    F.pad(state[0].view(state[0].shape[0], batch_size, self.beam_size, state[0].shape[-1]), (0, 0, 0, self.maes_expansion_delta), mode='constant').view(state[0].shape[0], -1, state[0].shape[-1]),
                    F.pad(state[1].view(state[0].shape[0], batch_size, self.beam_size, state[0].shape[-1]), (0, 0, 0, self.maes_expansion_delta), mode='constant').view(state[0].shape[0], -1, state[0].shape[-1]),
                )
                
                expansion_beam_idx = torch.arange(self.maes_expansion_max_hyps, device=device).unsqueeze(0).repeat(batch_size, 1)
                expansion_labels = torch.full((batch_size, self.maes_expansion_max_hyps, self.maes_num_steps), fill_value=-1, device=device)
                
                if self.ngram_lm_batch is not None:
                    lm_scores = F.pad(lm_scores, (0, 0, 0, self.maes_expansion_delta, 0, 0), mode='constant', value=float('-inf'))
                    batch_lm_states_candidates = F.pad(batch_lm_states_candidates.view(batch_size, self.beam_size, batch_lm_states_candidates.shape[-1]), (0, 0, 0, self.maes_expansion_delta), mode='constant').view(-1, batch_lm_states_candidates.shape[-1])
                    batch_lm_states = F.pad(batch_lm_states.view(batch_size, self.beam_size), (0, self.maes_expansion_delta), mode='constant').view(-1)
                
                expansion_steps=0
                num_expansions = self.maes_expansion_max_hyps
                while to_update.any() and expansion_steps < self.maes_num_steps:
                    expansion_beam_delta_indices = torch.arange(self.maes_expansion_max_hyps,
                                                                device=device).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, num_expansions)
                    
                    if self.pruning_mode is rnnt_utils.PruningMode.EARLY:
                        label_logps, labels = logps.topk(num_expansions, dim=-1, largest=True, sorted=True)
                        total_logps = expansion_logps.unsqueeze(-1) + label_logps
                        
                        # pruning with gamma
                        total_logps[total_logps <= total_logps.max(dim=-1, keepdim=True).values - self.maes_expansion_gamma] = float('-inf')
                        
                        if self.ngram_lm_batch is not None:
                            if self.blank_lm_score_mode is rnnt_utils.BlankLMScoreMode.NO_SCORE:
                                total_logps += torch.gather(lm_scores, dim=-1, index=labels)    
                            else:
                                raise NotImplementedError
                    else:
                        if self.ngram_lm_batch is not None:
                            if self.blank_lm_score_mode is rnnt_utils.BlankLMScoreMode.NO_SCORE:
                                logps = (logps + lm_scores)
                            elif self.blank_lm_score_mode is rnnt_utils.BlankLMScoreMode.LM_WEIGHTED_FULL:
                                blank_logprob = logps[..., -1]
                                non_blank_logprob = torch.log1p(-torch.clamp(torch.exp(blank_logprob), max=1.0 - 1e-6))
                                # assert (abs(torch.exp(blank_logprob) + torch.exp(non_blank_logprob) - 1.0) < 1e-5).all()

                                logps[..., :-1] += non_blank_logprob.unsqueeze(-1) * self.ngram_lm_alpha + lm_scores[..., :-1]
                                logps[..., -1] *= 1 + self.ngram_lm_alpha
                            else:
                                raise NotImplementedError
                                
                            label_logps, labels = logps.topk(num_expansions, dim=-1, largest=True, sorted=True)
                            
                            # pruning with gamma
                            total_logps = expansion_logps.unsqueeze(-1) + label_logps
                            total_logps[total_logps <= total_logps.max(dim=-1, keepdim=True).values - self.maes_expansion_gamma] = float('-inf')   
                    
                    total_logps = torch.where(to_update.unsqueeze(-1), total_logps, float('-inf'))
                    labels = torch.where(to_update.unsqueeze(-1), labels, -1)
                    
                    new_expansion_hashes = torch.where(labels != self._SOS, hash_text(expansion_hashes, labels), expansion_hashes)
                    masked_hashes = torch.where(batched_hyps.scores != float('-inf'), batched_hyps.hashes, -1)
                    init_expansions_equal = (new_expansion_hashes.view(batch_size, -1)[:, :, None] == masked_hashes[:, None, :]).any(dim=-1)
                    init_expansions_equal = torch.logical_and((labels != self._SOS).view(batch_size, -1), init_expansions_equal)
                    expansions_equal = new_expansion_hashes.view(batch_size, -1)[:, :, None] == new_expansion_hashes.view(batch_size, -1)[:, None, :]
                    expansion_scores = total_logps.view(batch_size, -1)
                    expansion_scores = torch.where(init_expansions_equal, float('-inf'), expansion_scores)
                    expansion_scores = expansion_scores[:, None, :].expand(expansions_equal.shape)
                    
                    expansion_scores = torch.where(expansions_equal, expansion_scores, float('-inf'))
                    expansion_scores, expansion_scores_argmax = expansion_scores.max(dim=-1)
                    scores_to_keep = (
                        torch.arange(expansion_scores_argmax.shape[-1], device=expansion_scores_argmax.device, dtype=torch.long)[None, :] == expansion_scores_argmax
                    )
                    total_logps = torch.where(scores_to_keep, expansion_scores, float('-inf')).view(batch_size, self.maes_expansion_max_hyps, -1)
                    
                    total_logps[..., -1] = torch.where(to_update, total_logps[..., -1], expansion_logps.squeeze(-1))
                    
                    expansion_logps, idx = total_logps.view(batch_size, -1).topk(self.maes_expansion_max_hyps, dim=-1, largest=True, sorted=True)
                    labels = labels.view(batch_size, -1)[batch_delta_indices, idx]
                    beam_idx = expansion_beam_delta_indices.view(batch_size, -1)[batch_delta_indices, idx]
                    expansion_hashes = new_expansion_hashes.view(batch_size, -1)[batch_delta_indices, idx].unsqueeze(-1)
                    
                    expansion_beam_idx = torch.gather(expansion_beam_idx, dim=1, index=beam_idx)
                    expansion_labels = torch.gather(expansion_labels, dim=1, index=beam_idx.unsqueeze(-1).repeat((1, 1, self.maes_num_steps)))
                    expansion_labels[expansion_label_indices == expansion_steps] = labels.flatten()
                    # expansion_labels = torch.where(expansion_label_indices == expansion_steps, labels, prev_expansion_labels)
                    
                    labels = torch.where(labels == -1, self._SOS, labels)
                    blank_mask = labels == self._SOS
                    
                    beam_state_idx = (batch_delta_indices * self.maes_expansion_max_hyps + beam_idx).flatten()
                    prev_state = self.decoder.batch_rearrange_states(state, beam_state_idx)
                    prev_decoder_output = torch.index_select(decoder_output, dim=0, index=beam_state_idx)
                    
                    decoder_output, state, *_ = self.decoder.predict(labels.view(-1, 1),
                                                                    prev_state,
                                                                    add_sos=False,
                                                                    batch_size=batch_size * self.maes_expansion_max_hyps)
                    decoder_output = self.joint.project_prednet(decoder_output)
                    
                    decoder_output = torch.where(blank_mask.flatten().unsqueeze(-1).unsqueeze(-1), prev_decoder_output, decoder_output)
                    state = (
                        torch.where(blank_mask.flatten().unsqueeze(0).unsqueeze(-1), prev_state[0], state[0]),
                        torch.where(blank_mask.flatten().unsqueeze(0).unsqueeze(-1), prev_state[1], state[1]),
                    )
                    
                    if self.ngram_lm_batch is not None:
                        batch_lm_states_candidates = torch.index_select(batch_lm_states_candidates, dim=0, index=beam_state_idx)
                        batch_lm_states_prev = torch.index_select(batch_lm_states, dim=0, index=beam_state_idx)
                        labels_w_blank_replaced = torch.where(blank_mask.flatten(), 0, labels.flatten())

                        batch_lm_states = torch.gather(batch_lm_states_candidates, dim=1, index=labels_w_blank_replaced.unsqueeze(-1)).flatten()
                        batch_lm_states = torch.where(blank_mask.flatten(), batch_lm_states_prev, batch_lm_states).view(-1)

                        lm_scores, batch_lm_states_candidates = self.ngram_lm_batch(
                            states=batch_lm_states
                        )
                        lm_scores = lm_scores.to(dtype=float_dtype).view(batch_size, self.maes_expansion_max_hyps, -1) * self.ngram_lm_alpha
                        lm_scores = torch.cat((lm_scores, zeros_column_delta), dim=2)
                    
                    logits = self.joint.joint_after_projection(encoder_output_projected[batch_delta_indices.flatten(), safe_time_indices.flatten()].unsqueeze(1), decoder_output)
                    logps = torch.log_softmax(logits, dim=-1).squeeze(1).squeeze(1).view(batch_size, self.maes_expansion_max_hyps, -1)
                    
                    beam_idx = torch.gather(expansion_beam_idx, dim=1, index=beam_idx)
                    to_update = torch.logical_and(to_update, labels != self._SOS)
                    
                    expansion_steps += 1
                    num_expansions = self.beam_size + self.maes_expansion_beta
                else:
                    # force blank expansion
                    expansion_logps = torch.where(to_update, expansion_logps + logps[..., -1], expansion_logps)
                    
                    expansion_logps, idx = expansion_logps.topk(self.beam_size, dim=-1, largest=True, sorted=True)
                    expansion_labels = torch.gather(expansion_labels, dim=1, index=idx.unsqueeze(-1).repeat(1, 1, self.maes_num_steps)).view(batch_size, self.beam_size, -1)
                    expansion_beam_idx = torch.gather(expansion_beam_idx, dim=1, index=idx).view(batch_size, self.beam_size)
                    expansion_logps = expansion_logps.view(batch_size, self.beam_size)
                    batched_hyps.update_beam(expansion_labels, expansion_logps, expansion_beam_idx)
                    
                    batch_idx = (batch_indices * self.maes_expansion_max_hyps + idx).flatten()
                    decoder_output = torch.index_select(decoder_output, dim=0, index=batch_idx)
                    state = (torch.index_select(state[0], dim=1, index=batch_idx),
                             torch.index_select(state[1], dim=1, index=batch_idx))
                
                    if self.ngram_lm_batch is not None:
                        batch_lm_states_candidates = torch.index_select(batch_lm_states_candidates, dim=0, index=batch_idx)
                        batch_lm_states = torch.index_select(batch_lm_states, dim=0, index=batch_idx)
                        
                        lm_scores = torch.index_select(lm_scores.view(batch_size * self.maes_expansion_max_hyps, -1), dim=0, index=batch_idx).view(batch_size, self.beam_size, -1)
                    
                time_indices += 1
                active_mask = time_indices <= last_timesteps
                
                step+=1

            return batched_hyps.get_best_hyps()

    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
    ) -> Tuple[rnnt_utils.BatchedHyps, Optional[rnnt_utils.BatchedAlignments], Any]:
        exact = True
        if exact == False:
            return self.batched_adaptive_expansion_search(encoder_output=x, encoder_output_length=out_len)
        else:
            return self.batched_adaptive_expansion_search_exact(encoder_output=x, encoder_output_length=out_len)