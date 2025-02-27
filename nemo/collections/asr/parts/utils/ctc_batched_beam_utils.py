import torch
from typing import Optional
from nemo.utils.enum import PrettyStrEnum
from collections import Counter

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

# https://stackoverflow.com/a/77213071
MULTIPLIER = 7
INCREMENT = 1
MODULUS = 2**64

def hash_text(prev_hash: torch.Tensor, add_labels: torch.Tensor) -> torch.Tensor:
    return prev_hash * MULTIPLIER + INCREMENT + add_labels

class CTCBatchedBeamHyps:
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
        self.INACTIVE_SCORE = -float("inf")
        self.INACTIVE_SCORE_TENSOR = torch.tensor(self.INACTIVE_SCORE, device=device, dtype=torch.float)
        self.INIT_POINTER_VALUE = -1
        self.INIT_PREFIX_HASH_VALUE = 0
        self.NON_EXISTENT_LABEL_VALUE = -1
                
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
            (batch_size, self.beam_size, self._max_length), fill_value=self.INIT_POINTER_VALUE, device=device, dtype=torch.long
        )
        self.last_label = torch.full([batch_size, self.beam_size], fill_value=self.NON_EXISTENT_LABEL_VALUE, device=device, dtype=torch.long)
        self.transcript_hash = torch.zeros([batch_size, self.beam_size], device=device, dtype=torch.long)
        self.timesteps = torch.zeros((batch_size, self.beam_size, self._max_length), device=device, dtype=torch.long)
        
        self.scores = torch.zeros([batch_size, self.beam_size], device=device, dtype=float_dtype)
        self.scores.fill_(self.INACTIVE_SCORE)
        self.scores[:, 0].fill_(0.0)

        self.batch_indices = torch.arange(self.batch_size, device=device)
        self.ZERO_TENSOR = torch.tensor(0, device=device, dtype=torch.long)  

    def clear_(self):
        self.current_lengths_nb.fill_(0)
        self.current_lengths_wb.fill_(0)
        self.last_label.fill_(self.NON_EXISTENT_LABEL_VALUE)
        self.timesteps.fill_(0)
        self.scores.fill_(self.INACTIVE_SCORE)
        self.scores[:, 0].fill_(0.0)
        
        self.transcript_wb.fill_(0)
        self.transcript_wb_prev_ptr.fill_(self.INIT_POINTER_VALUE)
        
        self.transcript_hash.fill_(0)

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
        # TODO: sdelat' chtom esli next_labels = -1, tut ne obnovlyalos nichego!
        self.scores.copy_(next_hyps_prob)
        self.transcript_wb.scatter_(dim=-1, index=self.current_lengths_wb.unsqueeze(-1), src=next_labels.unsqueeze(-1))
        self.transcript_wb_prev_ptr.scatter_(
            dim=-1, index=self.current_lengths_wb.unsqueeze(-1), src=hyps_indices.unsqueeze(-1)
        )
        # self.transcript.scatter_(dim=-1, index=self.current_lengths_nb.unsqueeze(-1), src=next_labels.unsqueeze(-1))
        # self.transcript_prev_ptr.scatter_(dim=-1, index=self.current_lengths_nb.unsqueeze(-1), src=hyps_indices.unsqueeze(-1))
        torch.add(self.current_lengths_wb, 1, out=self.current_lengths_wb)
        extended_with_blank = next_labels == self.blank_index
        extended_with_label = (~extended_with_blank) & (next_labels >= 0)
        self.current_lengths_nb.copy_(
            torch.gather(self.current_lengths_nb, dim=-1, index=hyps_indices) + extended_with_label
        )

        self.transcript_hash = torch.gather(self.transcript_hash, dim=-1, index=hyps_indices)
        mask_to_update_mask = torch.logical_and(next_labels != self.blank_index, next_labels != self.last_label)
        # update hashes and prefix hashes
        torch.where(
            mask_to_update_mask,
            hash_text(self.transcript_hash, next_labels),
            self.transcript_hash,
            out=self.transcript_hash
        )
        
        self.last_label.copy_(next_labels)
    
    def to_hyps_list(self, score_norm: bool = True) -> list[Hypothesis]:
        normalized_scores = self.scores / (self.current_lengths_nb.to(self.scores.dtype) + 1) if score_norm else self.scores
        _, best_hyp_index = torch.max(normalized_scores, dim=-1)
                
        scores = self.scores[self.batch_indices, best_hyp_index].tolist()
        
        tokens_list = []
        max_idx = self.current_lengths_wb.max() - 1
        ptr = best_hyp_index
        while max_idx >= 0:
            tokens = self.transcript_wb[self.batch_indices, ptr, max_idx]
            ptr = self.transcript_wb_prev_ptr[self.batch_indices, ptr, max_idx]

            max_idx -= 1
            tokens_list.insert(0, tokens)
        
        transcripts = torch.stack(tokens_list, dim=1).cpu().detach().numpy()
        hypotheses = [
            Hypothesis(
                score=scores[i],
                y_sequence=transcripts[i][transcripts[i] >= 0],
                timestamp=[],
                alignments=None,
                dec_state=None,
            )
            for i, _ in enumerate(range(self.batch_size))
        ]
        return hypotheses
    
    def self_recombine_hyps_(self):
        if self.beam_size <= 1:
            return
        # TODO: separate lm scores
        hyps_equal = (
            (self.transcript_hash[:, :, None] == self.transcript_hash[:, None, :])
            & (self.last_label[:, :, None] == self.last_label[:, None, :])
        )

        scores_matrix = torch.where(
            hyps_equal,
            self.scores[:, None, :].expand(self.batch_size, self.beam_size, self.beam_size),
            self.INACTIVE_SCORE_TENSOR,
        )
        scores_argmax = scores_matrix.argmax(-1, keepdim=False)
        scores_to_keep = (
            torch.arange(self.beam_size, device=scores_argmax.device, dtype=torch.long)[None, :] == scores_argmax
        )
        new_scores = torch.logsumexp(scores_matrix, dim=-1, keepdim=False)
        torch.where(scores_to_keep, new_scores.to(self.scores.dtype), self.INACTIVE_SCORE_TENSOR, out=self.scores)