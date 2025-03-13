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
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig

from nemo.collections.asr.parts.submodules.ngram_lm import FastNGramLM
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin
from nemo.collections.asr.parts.utils.rnnt_batched_beam_utils import BatchedBeamHyps, BlankLMScoreMode, PruningMode

class ModifiedAESBatchedRNNTComputer(ConfidenceMethodMixin):
    """
    Batched mAES decoding: https://ieeexplore.ieee.org/document/9250505
    """

    def __init__(
        self,
        decoder,
        joint,
        blank_index: int,
        beam_size: int,
        maes_num_steps: int,
        maes_expansion_beta: int,
        maes_expansion_gamma: int,
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
        self.maes_num_steps = maes_num_steps
        self.maes_expansion_beta = maes_expansion_beta
        self.maes_expansion_gamma = maes_expansion_gamma
        self.preserve_alignments = preserve_alignments
        self.preserve_frame_confidence = preserve_frame_confidence
        self.allow_recombine_hyps = allow_recombine_hyps
        self._SOS = self._blank_index
        self._init_confidence_method(confidence_method_cfg=confidence_method_cfg)
        self.score_norm = score_norm
        self.pruning_mode = pruning_mode
        self.blank_lm_score_mode = blank_lm_score_mode
        assert self._SOS == self._blank_index  # "blank as pad" algorithm only
        assert not self.preserve_alignments
        assert not self.preserve_frame_confidence
        
        if ngram_lm_model is not None:
            assert self._blank_index == self.joint.num_classes_with_blank - self.joint.num_extra_outputs - 1
            # self.ngram_lm_batch = FastNGramLM.from_arpa(lm_path=ngram_lm_model, vocab_size=self._blank_index)
            self.ngram_lm_batch = FastNGramLM.from_file(lm_path=ngram_lm_model, vocab_size=self._blank_index)
            
            self.pruning_mode = (
                PruningMode.EARLY
                if pruning_mode is None
                else PruningMode(pruning_mode)
            )
            self.blank_lm_score_mode = (
                BlankLMScoreMode.LM_WEIGHTED_FULL
                if blank_lm_score_mode is None 
                else BlankLMScoreMode(blank_lm_score_mode)
            )
        else:
            self.ngram_lm_batch = None
            self.blank_lm_score_mode = None
        self.ngram_lm_alpha = ngram_lm_alpha

    def batched_modified_adaptive_expansion_search_torch(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
    ) -> list[Hypothesis]:
        """
        Pure PyTorch implementation

        Args:
            encoder_output: output from the encoder
            encoder_output_length: lengths of the utterances in `encoder_output`
        """
        batch_size, max_time, _unused = encoder_output.shape
        device = encoder_output.device
        
        if self.ngram_lm_batch is not None:
            self.ngram_lm_batch.to(device)
        
        encoder_output_projected = self.joint.project_encoder(encoder_output)
        float_dtype = encoder_output_projected.dtype
        
        # init empty batched hypotheses
        batched_hyps = BatchedBeamHyps(
            batch_size=batch_size,
            beam_size=self.beam_size,
            blank_index=self._blank_index,
            init_length=max_time * (self.maes_num_steps + 1) if self.maes_num_steps is not None else max_time,
            device=device,
            float_dtype=float_dtype,
        )
        
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1).repeat(1, self.beam_size)
        expansion_beam_indices = torch.arange(self.beam_size, device=device).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, self.beam_size+self.maes_expansion_beta)
        beam_indices = torch.arange(self.beam_size, device=device).unsqueeze(0).repeat(batch_size, 1)
        
        time_indices = torch.zeros_like(batch_indices)
        safe_time_indices = torch.zeros_like(time_indices)
        last_timesteps = (encoder_output_length - 1).unsqueeze(-1).repeat(1, self.beam_size)
        active_mask = time_indices <= last_timesteps
        
        if self.ngram_lm_batch is not None:
            batch_lm_states = self.ngram_lm_batch.get_init_states(batch_size=batch_size * self.beam_size, bos=True)
            lm_scores, batch_lm_states_candidates = self.ngram_lm_batch.advance(states=batch_lm_states)  # vocab_size_no_blank
            lm_scores = lm_scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1) * self.ngram_lm_alpha
            # lm_scores = torch.cat((lm_scores, zeros_column), dim=2)
        
        prev_state = None
        last_labels_wb = torch.full(
            [batch_size, self.beam_size], fill_value=self._SOS, device=device, dtype=torch.long
        )
        decoder_output, state, *_ = self.decoder.predict(last_labels_wb.view(-1, 1),
                                                                 None,
                                                                 add_sos=False,
                                                                 batch_size=batch_size * self.beam_size)
        decoder_output = self.joint.project_prednet(decoder_output)
        step=0
        while active_mask.any():
            labels = batched_hyps.last_label
            to_update = active_mask.clone()
            safe_time_indices = torch.where(active_mask, time_indices, last_timesteps)
            
            logits = self.joint.joint_after_projection(
                encoder_output_projected[batch_indices.flatten(), safe_time_indices.flatten()].unsqueeze(1),
                decoder_output
            )
            logps = torch.log_softmax(logits, dim=-1).squeeze(1).squeeze(1).view(batch_size, self.beam_size, -1)
            
            updated_logps=self.combine_scores(logps, lm_scores) if self.ngram_lm_batch is not None else logps
            batched_hyps.recombine_prefixes(updated_logps, active_mask)
            
            expansion_steps=0
            while to_update.any() and expansion_steps < self.maes_num_steps:
                if self.ngram_lm_batch is None:
                    # choosing topk from acoustic model
                    label_logps, labels = logps.topk(self.beam_size + self.maes_expansion_beta, dim=-1, largest=True, sorted=True)
                
                    # pruning with gamma
                    total_logps = batched_hyps.scores.unsqueeze(-1) + label_logps
                    total_logps[total_logps <= total_logps.max(dim=-1, keepdim=True).values - self.maes_expansion_gamma] = float('-inf')
                    
                    labels = torch.where(to_update.unsqueeze(-1), labels, -1)
                    total_logps = torch.where(to_update.unsqueeze(-1), total_logps, batched_hyps.INACTIVE_SCORE)
                    
                    total_logps = batched_hyps.remove_duplicates(labels, total_logps)
                    total_logps[..., -1] = torch.where(to_update, total_logps[..., -1], batched_hyps.scores)
                                
                    total_logps, idx = total_logps.view(batch_size, -1).topk(self.beam_size, dim=-1, largest=True, sorted=True)
                    labels = labels.view(batch_size, -1)[batch_indices, idx]
                    beam_idx = expansion_beam_indices.view(batch_size, -1)[batch_indices, idx]
                else:
                    labels, total_logps, beam_idx = self.get_topk_lm(batch_size, batched_hyps, batch_indices, expansion_beam_indices, lm_scores, to_update, logps)
                
                batched_hyps.add_results_(beam_idx, labels, total_logps)
                
                labels = torch.where(labels == -1, self._SOS, labels)
                blank_mask = labels == self._SOS
                
                                
                beam_state_idx = (batch_indices * self.beam_size + beam_idx).flatten()
                prev_state = (
                    torch.index_select(state[0], dim=1, index=beam_state_idx.flatten()),
                    torch.index_select(state[1], dim=1, index=beam_state_idx.flatten())
                )
                prev_decoder_output = torch.index_select(decoder_output, dim=0, index=beam_state_idx)
                
                decoder_output, state, *_ = self.decoder.predict(labels.view(-1, 1),
                                                                 prev_state,
                                                                 add_sos=False,
                                                                 batch_size=batch_size * self.beam_size)
                decoder_output = self.joint.project_prednet(decoder_output)
                
                decoder_output = torch.where(
                    blank_mask.flatten().unsqueeze(-1).unsqueeze(-1), prev_decoder_output, decoder_output
                )
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

                    lm_scores, batch_lm_states_candidates = self.ngram_lm_batch.advance(states=batch_lm_states)
                    lm_scores = lm_scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1) * self.ngram_lm_alpha
                    # lm_scores = torch.cat((lm_scores, zeros_column), dim=2)
                
                logits = self.joint.joint_after_projection(
                    encoder_output_projected[batch_indices.flatten(),
                    safe_time_indices.flatten()].unsqueeze(1), 
                    decoder_output
                )
                logps = torch.log_softmax(logits, dim=-1).squeeze(1).squeeze(1).view(batch_size, self.beam_size, -1)
                
                to_update = torch.logical_and(to_update, labels != self._SOS)
                
                expansion_steps += 1
            else:
                if to_update.any():
                    # force blank expansion
                    total_logps = torch.where(to_update, batched_hyps.scores + logps[..., -1], batched_hyps.scores)
                    labels = torch.where(to_update, self._blank_index, -1)
                    batched_hyps.add_results_(beam_indices, labels, total_logps)
                        

            time_indices += 1
            active_mask = time_indices <= last_timesteps
            
            step+=1
        return batched_hyps.to_hyps_list(score_norm=self.score_norm)
    
    def combine_scores(self, log_probs, lm_scores):
        res=log_probs.clone()
        if self.blank_lm_score_mode is BlankLMScoreMode.NO_SCORE:
            # choosing topk from acoustic and Ngram models
            res[..., :-1] += lm_scores
        elif self.blank_lm_score_mode is BlankLMScoreMode.LM_WEIGHTED_FULL:
            blank_logprob = log_probs[..., -1]
            non_blank_logprob = torch.log1p(-torch.clamp(torch.exp(blank_logprob), max=1.0 - 1e-6))
            res[..., :-1] += non_blank_logprob.unsqueeze(-1) * self.ngram_lm_alpha + lm_scores
            res[..., -1] *= (1 + self.ngram_lm_alpha)
        else:
            raise NotImplementedError

        return res

    
    def get_topk_lm(self, batch_size, batched_hyps, batch_indices, expansion_beam_indices, lm_scores, to_update, log_probs):
        if self.pruning_mode is PruningMode.LATE:
            if self.blank_lm_score_mode in (BlankLMScoreMode.NO_SCORE, BlankLMScoreMode.LM_WEIGHTED_FULL):
                # choosing topk from acoustic and Ngram models
                log_probs = self.combine_scores(log_probs, lm_scores)
            else:
                raise NotImplementedError
            
            label_logps, labels = log_probs.topk(self.beam_size + self.maes_expansion_beta, dim=-1, largest=True, sorted=True)
                
            # pruning with gamma
            total_logps = batched_hyps.scores.unsqueeze(-1) + label_logps
            total_logps[total_logps <= total_logps.max(dim=-1, keepdim=True).values - self.maes_expansion_gamma] = float('-inf')

        elif self.pruning_mode is PruningMode.EARLY:
            if self.blank_lm_score_mode is BlankLMScoreMode.NO_SCORE:
                # choosing topk from acoustic model
                label_logps, labels = log_probs.topk(self.beam_size + self.maes_expansion_beta, dim=-1, largest=True, sorted=True)
                
                # pruning with gamma
                total_logps = batched_hyps.scores.unsqueeze(-1) + label_logps
                total_logps[total_logps <= total_logps.max(dim=-1, keepdim=True).values - self.maes_expansion_gamma] = float('-inf')
                
                # adding scores from ngram LM
                masked_labels = torch.where(labels==self._blank_index, 0, labels)
                total_logps = torch.where(
                    labels==self._blank_index,
                    total_logps,
                    total_logps + torch.gather(lm_scores, dim=-1,index=masked_labels))
            elif self.blank_lm_score_mode is BlankLMScoreMode.LM_WEIGHTED_FULL:
                # log_probs[..., :-1] += non_blank_logprob.unsqueeze(-1) * self.ngram_lm_alpha
                # choosing topk from acoustic model
                label_logps, labels = log_probs.topk(self.beam_size + self.maes_expansion_beta, dim=-1, largest=True, sorted=True)
                
                # pruning with gamma
                total_logps = batched_hyps.scores.unsqueeze(-1) + label_logps
                label_logps[total_logps <= total_logps.max(dim=-1, keepdim=True).values - self.maes_expansion_gamma] = float('-inf')
                
                blank_logprob = log_probs[..., -1]
                non_blank_logprob = torch.log1p(-torch.clamp(torch.exp(blank_logprob), max=1.0 - 1e-6))
                
                masked_labels = torch.where(labels==self._blank_index, 0, labels)
                total_logps = torch.where(
                    labels==self._blank_index,
                    total_logps + label_logps * (1 + self.ngram_lm_alpha),
                    total_logps + label_logps + non_blank_logprob.unsqueeze(-1) * self.ngram_lm_alpha + torch.gather(lm_scores, dim=-1, index=masked_labels)
                )
            else:
                raise NotImplementedError
        else:
                raise NotImplementedError
        
        labels = torch.where(to_update.unsqueeze(-1), labels, -1)
        total_logps = torch.where(to_update.unsqueeze(-1), total_logps, batched_hyps.INACTIVE_SCORE)
        
        total_logps = batched_hyps.remove_duplicate_new(labels, total_logps)
        total_logps[..., -1] = torch.where(to_update, total_logps[..., -1], batched_hyps.scores)
                    
        total_logps, idx = total_logps.view(batch_size, -1).topk(self.beam_size, dim=-1, largest=True, sorted=True)
        labels = labels.view(batch_size, -1)[batch_indices, idx]
        beam_idx = expansion_beam_indices.view(batch_size, -1)[batch_indices, idx]
        
        return labels, total_logps, beam_idx

    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
    ) -> list[Hypothesis]:
        return self.batched_modified_adaptive_expansion_search_torch(encoder_output=x, encoder_output_length=out_len)
