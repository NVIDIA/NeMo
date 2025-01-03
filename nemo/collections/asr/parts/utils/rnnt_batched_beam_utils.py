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
    # No score for blank
    NO_SCORE = "no_score"
    # If the blank label is in the top-k, keep it 
    # and select the top-k-1 labels based on their combined label and LM scores. 
    # Otherwise, select the top-k labels based on combined scores.
    PRESERVE_BLANK = "preserve_blank"
    # Blank score is obtained from Transducer model and weighted by LM weight.
    LM_WEIGHTED = "lm_weighted"
    
    LM_WEIGHTED_FULL = "lm_weighted_full"
    LM_WEIGHTED_FULL_FIXED_BLANK = "lm_weighted_full_fixed_blank"
    LM_MAX = "lm_max"
    LM_TOP_MAX = "lm_top_max"
    
class PruningMode(PrettyStrEnum):
    EARLY = "early"
    LATE = "late"
    
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
        self.INACTIVE_SCORE = -float("inf")
                
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
        self.scores.fill_(self.INACTIVE_SCORE)
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
        self.scores.fill_(self.INACTIVE_SCORE)
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
            0,
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
            self.INACTIVE_SCORE,
        )
        scores_argmax = scores_matrix.argmax(-1, keepdim=False)
        scores_to_keep = (
            torch.arange(self.beam_size, device=scores_argmax.device, dtype=torch.long)[None, :] == scores_argmax
        )
        new_scores = torch.logsumexp(scores_matrix, dim=-1, keepdim=False)
        torch.where(scores_to_keep, new_scores, torch.tensor(self.INACTIVE_SCORE), out=self.scores)

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