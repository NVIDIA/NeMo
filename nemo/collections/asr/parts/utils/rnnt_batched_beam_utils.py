import torch
from typing import Optional
from nemo.utils.enum import PrettyStrEnum

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

# https://stackoverflow.com/a/77213071
MULTIPLIER = 6364136223846793005
INCREMENT = 1
MODULUS = 2**64

def hash_text(prev_hash: torch.Tensor, add_labels: torch.Tensor) -> torch.Tensor:
    return prev_hash * MULTIPLIER + INCREMENT + add_labels

class BlankLMScoreMode(PrettyStrEnum):
    """
        Defines the strategies for handling blank token scores in a external Ngram LM
        when combined with an automatic speech recognition (ASR) model.
    """
    NO_SCORE = "no_score"
    """No score for blank."""
    LM_WEIGHTED_FULL = "lm_weighted_full"
    """Blank score for LM is set equal to blank score from ASR model; non-blank LM scores are reweighted to sum to 1."""  

class PruningMode(PrettyStrEnum):
    """Specifies when pruning is applied external Ngram LM shallow fusion.."""
    EARLY = "early"
    """Hyps are pruned based on ASR probs, then rescored with LM"""
    LATE = "late"
    """Hyps are scored based on combined ASR and LM probs., then pruned"""
    
class BatchedBeamHyps:
    """Class to store batch of beam hypotheses (labels, time_indices, scores) for efficient RNNT decoding"""

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
        self.transcript_hash = torch.zeros([batch_size, self.beam_size], device=device, dtype=torch.long)
        self.transcript_prefix_hash = torch.full([batch_size, self.beam_size], device=device, dtype=torch.long, fill_value=self.INIT_PREFIX_HASH_VALUE)
        self.last_label = torch.full([batch_size, self.beam_size], fill_value=self.NON_EXISTENT_LABEL_VALUE, device=device, dtype=torch.long)
        self.timesteps = torch.zeros((batch_size, self.beam_size, self._max_length), device=device, dtype=torch.long)
        
        self.scores = torch.zeros([batch_size, self.beam_size], device=device, dtype=float_dtype)
        self.scores.fill_(self.INACTIVE_SCORE)
        self.scores[:, 0].fill_(0.0)

        self.next_timestep = torch.zeros((batch_size, self.beam_size), device=device, dtype=torch.long)
        self.last_timestep_lasts = torch.zeros((batch_size, self.beam_size), device=device, dtype=torch.long)
        
        self.batch_indices = torch.arange(self.batch_size, device=device)

    def clear_(self):
        self.current_lengths_nb.fill_(0)
        self.current_lengths_wb.fill_(0)
        self.last_label.fill_(self.NON_EXISTENT_LABEL_VALUE)
        self.timesteps.fill_(0)
        self.scores.fill_(self.INACTIVE_SCORE)
        self.scores[:, 0].fill_(0.0)
        self.next_timestep.fill_(0)
        self.last_timestep_lasts.fill_(0)
        
        self.transcript_wb.fill_(0)
        self.transcript_wb_prev_ptr.fill_(self.INIT_POINTER_VALUE)
        
        self.transcript_hash.fill_(0)
        self.transcript_prefix_hash.fill_(self.INIT_PREFIX_HASH_VALUE)

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
        torch.add(self.current_lengths_wb, 1, out=self.current_lengths_wb)
        extended_with_blank = next_labels == self.blank_index
        extended_with_label = (~extended_with_blank) & (next_labels >= 0)
        self.current_lengths_nb.copy_(
            torch.gather(self.current_lengths_nb, dim=-1, index=hyps_indices) + extended_with_label
        )

        self.next_timestep.copy_(self.current_lengths_wb - self.current_lengths_nb)
        self.last_timestep_lasts.copy_(torch.where(
            extended_with_blank,
            0,
            torch.gather(self.last_timestep_lasts, dim=-1, index=hyps_indices) + extended_with_label,
        ))

        prev_transcript_hash = torch.gather(self.transcript_hash, dim=-1, index=hyps_indices)
        prev_transcript_prefix_hash = torch.gather(self.transcript_prefix_hash, dim=-1, index=hyps_indices)
        last_labels=torch.gather(self.last_label, dim=-1, index=hyps_indices)
        # track last label
        torch.where(
            extended_with_label,
            next_labels,
            last_labels,
            out=self.last_label,
        )
        
        # update hashes and prefix hashes
        torch.where(
            extended_with_label,
            hash_text(prev_transcript_hash, next_labels),
            prev_transcript_hash,
            out=self.transcript_hash
        )
        torch.where(
            extended_with_label,
            prev_transcript_hash,
            prev_transcript_prefix_hash,
            out=self.transcript_prefix_hash
        )

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
            self.INACTIVE_SCORE_TENSOR,
        )
        scores_argmax = scores_matrix.argmax(-1, keepdim=False)
        scores_to_keep = (
            torch.arange(self.beam_size, device=scores_argmax.device, dtype=torch.long)[None, :] == scores_argmax
        )
        new_scores = torch.logsumexp(scores_matrix, dim=-1, keepdim=False)
        torch.where(scores_to_keep, new_scores.to(self.scores.dtype), self.INACTIVE_SCORE_TENSOR, out=self.scores)
    
    def remove_duplicates(self, labels, total_logps):
        if self.beam_size <= 1:
            return total_logps
        
        non_blanks_mask = labels != self.blank_index
        expansion_number = labels.view(self.batch_size, -1).shape[-1]
                
        # expansions
        expansion_hashes = hash_text(self.transcript_hash.unsqueeze(-1), labels)
        expansion_hashes = torch.where(non_blanks_mask, expansion_hashes, self.transcript_hash.unsqueeze(-1))
        expansions_equal = expansion_hashes.view(self.batch_size, -1)[:, :, None] == expansion_hashes.view(self.batch_size, -1)[:, None, :]
        expansion_scores = total_logps.view(self.batch_size, -1)
        expansion_scores = expansion_scores[:, None, :].expand((self.batch_size, expansion_number, expansion_number))
        
        expansion_scores = torch.where(expansions_equal, expansion_scores, float('-inf'))
        expansion_scores, expansion_scores_argmax = expansion_scores.max(dim=-1)
        scores_to_keep = (
            torch.arange(expansion_number, device=expansion_scores_argmax.device, dtype=torch.long)[None, :] == expansion_scores_argmax
        )
        recombined_logps = torch.where(scores_to_keep, expansion_scores, float('-inf'))

        return recombined_logps.view(self.batch_size, self.beam_size, -1)
    
    def remove_duplicate_new(self, labels, total_logps):
        if self.beam_size <= 1:
            return total_logps
        
        # updating hashes for label expansions
        non_blank_mask = labels != self.blank_index
        expansion_hashes=hash_text(self.transcript_hash.unsqueeze(-1), labels)
        expansion_hashes = torch.where(non_blank_mask, expansion_hashes, self.transcript_hash.unsqueeze(-1)).view(self.batch_size, -1)
        
        # masking inactive hypotheses
        inactive_hyps_mask = self.scores != self.INACTIVE_SCORE
        masked_hashes = torch.where(inactive_hyps_mask, self.transcript_hash, -1)
        
        init_expansions_equal = (
            expansion_hashes[:, :, None] == masked_hashes[:, None, :]
            ).any(dim=-1)
        
        init_expansions_equal = torch.logical_and(non_blank_mask.view(self.batch_size, -1), init_expansions_equal)
        expansions_equal = expansion_hashes[:, :, None] == expansion_hashes[:, None, :]
        expansion_scores = total_logps.view(self.batch_size, -1)
        expansion_scores = torch.where(init_expansions_equal, self.INACTIVE_SCORE, expansion_scores)
        expansion_scores = expansion_scores[:, None, :].expand(expansions_equal.shape)
        
        expansion_scores = torch.where(expansions_equal, expansion_scores, self.INACTIVE_SCORE)
        expansion_scores, expansion_scores_argmax = expansion_scores.max(dim=-1)
        
        scores_range = torch.arange(
            expansion_scores_argmax.shape[-1],
            device=expansion_scores_argmax.device,
            dtype=torch.long
        )
        scores_to_keep = scores_range[None, :] == expansion_scores_argmax
        total_logps = torch.where(scores_to_keep, expansion_scores, self.INACTIVE_SCORE).view(self.batch_size, self.beam_size, -1)

        return total_logps

      
    def recombine_prefixes(self, label_logps: torch.Tensor, active_mask: torch.Tensor):
        if self.beam_size <= 1:
            return
        
        # if hypotheses are empty skip
        if (self.current_lengths_wb == 0).any():
            return
        
        # mask prefix hashes if hypotheses of the beam do not have prefixes (e.g. no non-blank labels were appended)
        prefix_hashes = torch.where(self.current_lengths_nb == 0, -2, self.transcript_prefix_hash)
        
        prefix_equal = self.transcript_hash[:, None, :] == prefix_hashes[:, :, None]
        
        last_labels=torch.where(self.last_label == self.NON_EXISTENT_LABEL_VALUE, self.blank_index, self.last_label)
        prefix_labels = last_labels.unsqueeze(1).repeat((1, self.beam_size, 1))
        prefix_scores = self.scores.unsqueeze(1).repeat((1, self.beam_size, 1))
        
        prefix_label_logps = torch.gather(label_logps, dim=-1, index=prefix_labels)
        prefix_label_logps = prefix_scores + prefix_label_logps.transpose(dim0=-1, dim1=-2)
        prefix_label_logps = torch.where(prefix_equal, prefix_label_logps, self.INACTIVE_SCORE)
        prefix_label_logps = torch.logsumexp(prefix_label_logps, dim=-1)

        to_update_mask = torch.logical_and(active_mask, self.scores != self.INACTIVE_SCORE)
        self.scores = torch.where(to_update_mask, torch.logaddexp(self.scores, prefix_label_logps), self.scores)


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
            self.INACTIVE_SCORE,
        )
        scores_argmax = scores_matrix.argmax(-1, keepdim=False)
        scores_to_keep = (
            torch.arange(self.beam_size * self.beam_size, device=device, dtype=torch.long)[None, :] == scores_argmax
        )
        scores_to_copy = (hyps_equal.sum(-1) == 1) | torch.isinf(hyps_extenstions_probs)
        new_scores = torch.logsumexp(scores_matrix, dim=-1, keepdim=False)
        # assert (~torch.isnan(new_scores)).all()
        scores = torch.where(scores_to_keep, new_scores, self.INACTIVE_SCORE)
        scores = torch.where(scores_to_copy, hyps_extenstions_probs, scores)
        return scores.view(self.batch_size, self.beam_size, self.beam_size)

    def to_hyps_list(self, score_norm: bool = True) -> list[Hypothesis]:
        normalized_scores = self.scores / (self.current_lengths_nb.to(self.scores.dtype) + 1) if score_norm else self.scores
        _, best_hyp_index = torch.max(normalized_scores, dim=-1)
                
        scores = self.scores[self.batch_indices, best_hyp_index].tolist()
        
        max_idx = self.current_lengths_wb.max() - 1
        tokens_list = []
        ptr = best_hyp_index
        for idx in range(max_idx, -1, -1):
            tokens = self.transcript_wb[self.batch_indices, ptr, idx]
            ptr = self.transcript_wb_prev_ptr[self.batch_indices, ptr, idx]

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
    
    
class BatchedBeamHypsTDT:
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
        self.device=device
        self.INACTIVE_SCORE = -float("inf")
        self.INACTIVE_SCORE_TENSOR = torch.tensor(self.INACTIVE_SCORE, device=device, dtype=torch.float)
        self.INIT_POINTER_VALUE = -1
        self.INIT_PREFIX_HASH_VALUE = 0
        self.NON_EXISTENT_LABEL_VALUE = -1
        self.ZERO_TENSOR = torch.tensor(0, device=device, dtype=torch.long)
                
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
        self.transcript_hash = torch.zeros([batch_size, self.beam_size], device=device, dtype=torch.long)
        self.transcript_prefix_hash = torch.full([batch_size, self.beam_size], device=device, dtype=torch.long, fill_value=self.INIT_PREFIX_HASH_VALUE)
        self.last_label = torch.full([batch_size, self.beam_size], fill_value=self.NON_EXISTENT_LABEL_VALUE, device=device, dtype=torch.long)
        self.timesteps = torch.zeros((batch_size, self.beam_size, self._max_length), device=device, dtype=torch.long)
        
        self.scores = torch.zeros([batch_size, self.beam_size], device=device, dtype=float_dtype)
        self.scores.fill_(self.INACTIVE_SCORE)
        self.scores[:, 0].fill_(0.0)

        self.next_timestep = torch.zeros((batch_size, self.beam_size), device=device, dtype=torch.long)
        self.last_timestep_lasts = torch.zeros((batch_size, self.beam_size), device=device, dtype=torch.long)

    def clear_(self):
        self.current_lengths_nb.fill_(0)
        self.current_lengths_wb.fill_(0)
        self.last_label.fill_(self.NON_EXISTENT_LABEL_VALUE)
        self.timesteps.fill_(0)
        self.scores.fill_(self.INACTIVE_SCORE)
        self.scores[:, 0].fill_(0.0)
        self.next_timestep.fill_(0)
        self.last_timestep_lasts.fill_(0)
        
        self.transcript_wb.fill_(0)
        self.transcript_wb_prev_ptr.fill_(self.INIT_POINTER_VALUE)
        
        self.transcript_hash.fill_(0)
        self.transcript_prefix_hash.fill_(self.INIT_PREFIX_HASH_VALUE)

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
        next_label_durations,
    ):
        if (self.current_lengths_wb + 1).max() >= self._max_length:
            self._allocate_more()
            
        self.add_results_no_checks_(
            hyps_indices=hyps_indices,
            next_labels=next_labels,
            next_hyps_prob=next_hyps_prob,
            next_label_durations=next_label_durations
        )

    def add_results_no_checks_(
        self,
        hyps_indices,
        next_labels,
        next_hyps_prob,
        next_label_durations
    ):
        # TODO: timesteps
        self.scores.copy_(next_hyps_prob)
        self.transcript_wb.scatter_(dim=-1, index=self.current_lengths_wb.unsqueeze(-1), src=next_labels.unsqueeze(-1))
        self.transcript_wb_prev_ptr.scatter_(
            dim=-1, index=self.current_lengths_wb.unsqueeze(-1), src=hyps_indices.unsqueeze(-1)
        )


        torch.add(self.current_lengths_wb, 1, out=self.current_lengths_wb)
        extended_with_blank = next_labels == self.blank_index
        extended_with_label = (~extended_with_blank) & (next_labels >= 0)
        self.current_lengths_nb.copy_(
            torch.gather(self.current_lengths_nb, dim=-1, index=hyps_indices) + extended_with_label
        )

        timesteps = torch.gather(self.next_timestep, dim=-1, index=hyps_indices)
        torch.where(next_labels >= 0, timesteps + next_label_durations, timesteps, out=self.next_timestep)
        torch.where(
            next_label_durations>0,
            self.ZERO_TENSOR,
            torch.gather(self.last_timestep_lasts, dim=-1, index=hyps_indices) + extended_with_label,
            out=self.last_timestep_lasts
        )

        prev_transcript_hash = torch.gather(self.transcript_hash, dim=-1, index=hyps_indices)
        prev_transcript_prefix_hash = torch.gather(self.transcript_prefix_hash, dim=-1, index=hyps_indices)
        last_labels=torch.gather(self.last_label, dim=-1, index=hyps_indices)
        # track last label
        torch.where(
            extended_with_label,
            next_labels,
            last_labels,
            out=self.last_label,
        )
        
        # update hashes and prefix hashes
        torch.where(
            extended_with_label,
            hash_text(prev_transcript_hash, next_labels),
            prev_transcript_hash,
            out=self.transcript_hash
        )
        torch.where(
            extended_with_label,
            prev_transcript_hash,
            prev_transcript_prefix_hash,
            out=self.transcript_prefix_hash
        )

    def self_recombine_hyps_(self):
        if self.beam_size <= 1:
            return
        # TODO: separate lm scores
        hyps_equal = (
            (self.transcript_hash[:, :, None] == self.transcript_hash[:, None, :])
            & (self.last_label[:, :, None] == self.last_label[:, None, :])
            & (self.current_lengths_nb[:, :, None] == self.current_lengths_nb[:, None, :])
            & (self.next_timestep[:, :, None] == self.next_timestep[:, None, :])
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
        torch.where(scores_to_keep, new_scores, self.INACTIVE_SCORE_TENSOR, out=self.scores)
    
    def remove_duplicates(self, labels, total_logps):
        if self.beam_size <= 1:
            return total_logps
        
        non_blanks_mask = labels != self.blank_index
        expansion_number = labels.view(self.batch_size, -1).shape[-1]
                
        # expansions
        expansion_hashes = hash_text(self.transcript_hash.unsqueeze(-1), labels)
        expansion_hashes = torch.where(non_blanks_mask, expansion_hashes, self.transcript_hash.unsqueeze(-1))
        expansions_equal = expansion_hashes.view(self.batch_size, -1)[:, :, None] == expansion_hashes.view(self.batch_size, -1)[:, None, :]
        expansion_scores = total_logps.view(self.batch_size, -1)
        expansion_scores = expansion_scores[:, None, :].expand((self.batch_size, expansion_number, expansion_number))
        
        expansion_scores = torch.where(expansions_equal, expansion_scores, float('-inf'))
        expansion_scores, expansion_scores_argmax = expansion_scores.max(dim=-1)
        scores_to_keep = (
            torch.arange(expansion_number, device=expansion_scores_argmax.device, dtype=torch.long)[None, :] == expansion_scores_argmax
        )
        recombined_logps = torch.where(scores_to_keep, expansion_scores, float('-inf'))

        return recombined_logps.view(self.batch_size, self.beam_size, -1)
    
    def remove_duplicate_new(self, labels, total_logps):
        if self.beam_size <= 1:
            return total_logps
        
        # updating hashes for label expansions
        non_blank_mask = labels != self.blank_index
        expansion_hashes=hash_text(self.transcript_hash.unsqueeze(-1), labels)
        expansion_hashes = torch.where(non_blank_mask, expansion_hashes, self.transcript_hash.unsqueeze(-1)).view(self.batch_size, -1)
        
        # masking inactive hypotheses
        inactive_hyps_mask = self.scores != self.INACTIVE_SCORE
        masked_hashes = torch.where(inactive_hyps_mask, self.transcript_hash, -1)
        
        init_expansions_equal = (
            expansion_hashes[:, :, None] == masked_hashes[:, None, :]
            ).any(dim=-1)
        
        init_expansions_equal = torch.logical_and(non_blank_mask.view(self.batch_size, -1), init_expansions_equal)
        expansions_equal = expansion_hashes[:, :, None] == expansion_hashes[:, None, :]
        expansion_scores = total_logps.view(self.batch_size, -1)
        expansion_scores = torch.where(init_expansions_equal, self.INACTIVE_SCORE, expansion_scores)
        expansion_scores = expansion_scores[:, None, :].expand(expansions_equal.shape)
        
        expansion_scores = torch.where(expansions_equal, expansion_scores, self.INACTIVE_SCORE)
        expansion_scores, expansion_scores_argmax = expansion_scores.max(dim=-1)
        
        scores_range = torch.arange(
            expansion_scores_argmax.shape[-1],
            device=expansion_scores_argmax.device,
            dtype=torch.long
        )
        scores_to_keep = scores_range[None, :] == expansion_scores_argmax
        total_logps = torch.where(scores_to_keep, expansion_scores, self.INACTIVE_SCORE).view(self.batch_size, self.beam_size, -1)

        return total_logps

      
    def recombine_prefixes(self, label_logps: torch.Tensor, active_mask: torch.Tensor):
        if self.beam_size <= 1:
            return
        
        # if hypotheses are empty skip
        if (self.current_lengths_wb == 0).any():
            return
        
        # mask prefix hashes if hypotheses of the beam do not have prefixes (e.g. no non-blank labels were appended)
        prefix_hashes = torch.where(self.current_lengths_nb == 0, -2, self.transcript_prefix_hash)
        
        prefix_equal = self.transcript_hash[:, None, :] == prefix_hashes[:, :, None]
        
        last_labels=torch.where(self.last_label == self.NON_EXISTENT_LABEL_VALUE, self.blank_index, self.last_label)
        prefix_labels = last_labels.unsqueeze(1).repeat((1, self.beam_size, 1))
        prefix_scores = self.scores.unsqueeze(1).repeat((1, self.beam_size, 1))
        
        prefix_label_logps = torch.gather(label_logps, dim=-1, index=prefix_labels)
        prefix_label_logps = prefix_scores + prefix_label_logps.transpose(dim0=-1, dim1=-2)
        prefix_label_logps = torch.where(prefix_equal, prefix_label_logps, self.INACTIVE_SCORE)
        prefix_label_logps = torch.logsumexp(prefix_label_logps, dim=-1)

        to_update_mask = torch.logical_and(active_mask, self.scores != self.INACTIVE_SCORE)
        self.scores = torch.where(to_update_mask, torch.logaddexp(self.scores, prefix_label_logps), self.scores)


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
            self.INACTIVE_SCORE,
        )
        scores_argmax = scores_matrix.argmax(-1, keepdim=False)
        scores_to_keep = (
            torch.arange(self.beam_size * self.beam_size, device=device, dtype=torch.long)[None, :] == scores_argmax
        )
        scores_to_copy = (hyps_equal.sum(-1) == 1) | torch.isinf(hyps_extenstions_probs)
        new_scores = torch.logsumexp(scores_matrix, dim=-1, keepdim=False)
        # assert (~torch.isnan(new_scores)).all()
        scores = torch.where(scores_to_keep, new_scores, self.INACTIVE_SCORE)
        scores = torch.where(scores_to_copy, hyps_extenstions_probs, scores)
        return scores.view(self.batch_size, self.beam_size, self.beam_size)

    def to_hyps_list(self, score_norm: bool = True) -> list[Hypothesis]:
        # self.batch_beam_indices=torch.arange(self.beam_size, device=self.device, dtype=torch.long)[None, :].expand(self.batch_size, -1)
        transcript = self.transcript_wb[..., :self.current_lengths_wb.max()].tolist()
        transcript_wb_prev_ptr = self.transcript_wb_prev_ptr[..., :self.current_lengths_wb.max()].tolist()
        if score_norm:
            end_indices = torch.argmax(self.scores / self.current_lengths_nb.to(self.scores.dtype), dim=-1).tolist()
        else:
            end_indices = torch.argmax(self.scores, dim=-1).tolist()
        scores = self.scores.tolist()
        batch_size = self.scores.shape[0]
        hyp_length = self.current_lengths_wb[0, 0].cpu().item()
        # TODO: faster parallel aggregation
        # TODO: timesteps
        hypotheses: list[Hypothesis] = []
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
                Hypothesis(
                    score=scores[batch_i][end_indices[batch_i]],
                    y_sequence=cur_transcript[::-1],
                    timestep=[],
                    alignments=None,
                    dec_state=None,
                )
            )
        return hypotheses