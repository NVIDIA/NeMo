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

import math
from collections import OrderedDict

from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.utils import logging
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import EncodedRepresentation, LengthsType, NeuralType, SpectrogramType
from nemo.core.neural_types.elements import ProbsType
from typing import Optional, Dict, List, Union, Tuple

__all__ = ['SortformerModules']


class SortformerModules(NeuralModule, Exportable):
    """
    A class including auxiliary functions for Sortformer models.
    This class contains and will contain the following functions that performs streaming features,
    and any neural layers that are not included in the NeMo neural modules (e.g. Transformer, Fast-Conformer).
    """

    def init_weights(self, m):
        """Init weights for linear layers."""
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(
        self,
        num_spks: int = 4,
        dropout_rate: float = 0.5,
        fc_d_model: int = 512,
        tf_d_model: int = 192,
        subsampling_factor: int = 8,
        mem_len: int = 188,
        fifo_len: int = 0,
        step_len: int = 376,
        mem_refresh_rate: int = 1,
        step_left_context: int = 0,
        step_right_context: int = 0,
        mem_sil_frames_per_spk: int = 5,
        causal_attn_rate: float = 0,
        causal_attn_rc: int = 7,
        use_causal_eval: bool = False,
        scores_add_rnd: float = 0,
        init_step_len: int = 999,
        pred_score_threshold: float = 0.25,
        max_index: int = 99999,
        max_score: float = 10.0,
        scores_boost_latest: float = 0.05,
        sil_threshold: float = 0.2,
        strong_boost_rate: float = 0.75,
        weak_boost_rate: float = 1.5,
        min_high_scores_rate: float = 0.5,
    ):
        super().__init__()
        self.mem_sil_frames_per_spk = mem_sil_frames_per_spk
        self.step_left_context = step_left_context
        self.step_right_context = step_right_context
        self.subsampling_factor = subsampling_factor
        self.mem_len = mem_len
        self.fifo_len = fifo_len
        self.step_len = step_len
        self.mem_refresh_rate = mem_refresh_rate
        self.fc_d_model = fc_d_model
        self.tf_d_model = tf_d_model
        self.hidden_size = tf_d_model
        self.unit_n_spks: int = num_spks
        self.hidden_to_spks = nn.Linear(2 * self.hidden_size, self.unit_n_spks)
        self.first_hidden_to_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.single_hidden_to_spks = nn.Linear(self.hidden_size, self.unit_n_spks)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_proj = nn.Linear(self.fc_d_model, self.tf_d_model)
        self.visualization: bool = False
        self.log = False
        self.causal_attn_rate = causal_attn_rate
        self.causal_attn_rc = causal_attn_rc
        self.use_causal_eval = use_causal_eval
        self.scores_add_rnd = scores_add_rnd
        self.init_step_len = init_step_len
        self.max_index = max_index
        self.max_score = max_score
        self.pred_score_threshold = pred_score_threshold
        self.scores_boost_latest = scores_boost_latest
        self.sil_threshold = sil_threshold
        self.strong_boost_rate = strong_boost_rate
        self.weak_boost_rate = weak_boost_rate
        self.min_high_scores_rate = min_high_scores_rate

    def length_to_mask(self, lengths, max_length: int):
        """
        Convert length values to encoder mask input tensor

        Args:
            lengths (torch.Tensor): tensor containing lengths of sequences
            max_length (int): maximum sequence length

        Returns:
            mask (torch.Tensor): tensor of shape (batch_size, max_len) containing 0's
                                in the padded region and 1's elsewhere
        """
        batch_size = lengths.shape[0]
        arange = torch.arange(max_length, device=lengths.device)
        mask = arange.expand(batch_size, max_length) < lengths.unsqueeze(1)
        return mask

    def streaming_feat_loader(self, feat_seq, feat_seq_length, feat_seq_offset) -> Tuple[int, torch.Tensor, torch.Tensor, int, int]:
        """
        Load a chunk of feature sequence for streaming inference.

        Args:
            feat_seq (torch.Tensor): Tensor containing feature sequence
                Dimension: (batch_size, feat_dim, feat frame count)
            feat_seq_length (torch.Tensor): Tensor containing feature sequence lengths
                Dimension: (batch_size,)
            feat_seq_offset (torch.Tensor): Tensor containing feature sequence offsets
                Dimension: (batch_size,)

        Returns:
            step_idx (int): Index of the current step
            chunk_feat_seq (torch.Tensor): Tensor containing the chunk of feature sequence
                Dimension: (batch_size, diar frame count, feat_dim)
            feat_lengths (torch.Tensor): Tensor containing lengths of the chunk of feature sequence
                Dimension: (batch_size,)
        """
        feat_len = feat_seq.shape[2]
        num_chunks = math.ceil(feat_len / (self.step_len * self.subsampling_factor))
        if self.log:
            logging.info(f"feat_len={feat_len}, num_chunks={num_chunks}, feat_seq_length={feat_seq_length}, feat_seq_offset={feat_seq_offset}")
        stt_feat, end_feat, step_idx = 0, 0, 0
        current_step_len = min(self.init_step_len, self.step_len)
        while end_feat < feat_len:
            left_offset = min(self.step_left_context * self.subsampling_factor, stt_feat)
            end_feat = min(stt_feat + current_step_len * self.subsampling_factor, feat_len)
            right_offset = min(self.step_right_context * self.subsampling_factor, feat_len-end_feat)
            chunk_feat_seq = feat_seq[:, :, stt_feat-left_offset:end_feat+right_offset]
            feat_lengths = (feat_seq_length + feat_seq_offset - stt_feat + left_offset).clamp(0,chunk_feat_seq.shape[2])
            feat_lengths = feat_lengths * (feat_seq_offset < end_feat)
            stt_feat = end_feat
            chunk_feat_seq_t = torch.transpose(chunk_feat_seq, 1, 2)
            if self.log:
                logging.info(f"step_idx: {step_idx}, current step len: {current_step_len}, chunk_feat_seq_t shape: {chunk_feat_seq_t.shape}, chunk_feat_lengths: {feat_lengths}")
            yield step_idx, chunk_feat_seq_t, feat_lengths, left_offset, right_offset
            step_idx += 1
            if end_feat >= self.step_len * self.subsampling_factor:
                current_step_len = self.step_len
            elif end_feat < self.step_len * self.subsampling_factor and current_step_len < self.step_len and end_feat >= 4*current_step_len*self.subsampling_factor:
                current_step_len *= 2
     
    def forward_speaker_sigmoids(self, hidden_out):
        """ 
        The final layer that renders speaker probabilities in Sigmoid activation function.
        
        Args:
            hidden_out (torch.Tensor): tensor containing hidden states from the encoder
                Dimension: (batch_size, max_len, hidden_size)
                
        Returns:
            preds (torch.Tensor): tensor containing speaker probabilities in Sigmoid activation function
                Dimension: (batch_size, num_spks)
        """
        hidden_out = self.dropout(F.relu(hidden_out))
        hidden_out = self.first_hidden_to_hidden(hidden_out)
        hidden_out = self.dropout(F.relu(hidden_out))
        spk_preds = self.single_hidden_to_spks(hidden_out)
        preds = nn.Sigmoid()(spk_preds)
        return preds

    def concat_embs(self, list_of_tensors=List[torch.Tensor], return_lengths: bool=False, dim: int=1, device: torch.device=None):
        """
        Concatenate a list of tensors along the specified dimension.

        Args:
            list_of_tensors (List[torch.Tensor]): List of tensors to concatenate
            return_lengths (bool): Whether to return lengths of the concatenated tensors
            dim (int): Dimension along which to concatenate
            device (torch.device): device to use for tensor operations

        Returns:
            embs (torch.Tensor): concatenated tensor
        """
        embs = torch.cat(list_of_tensors, dim=dim).to(device) # Shape: (batch_size, frames (max_len), total_emb_dim)
        lengths = torch.tensor(embs.shape[1]).repeat(embs.shape[0]).to(device)

        if return_lengths:
            return embs, lengths
        else:
            return embs

    def init_memory(self, batch_size: int, d_model: int, device: torch.device):
        return torch.zeros(batch_size, 0, d_model).to(device)

    def update_memory_fifo_async(
        self,
        mem,
        mem_lengths,
        mem_preds,
        fifo,
        fifo_lengths,
        chunk,
        chunk_lengths,
        preds,
        lc: int = 0,
        rc: int = 0
    ):
        """
        update the FIFO queue and memory buffer with the chunk of embeddings and speaker predictions
        Args:
            mem (torch.Tensor): memory buffer to save the embeddings from start
                Dimension: (batch_size, mem_len, emb_dim)
            mem_lengths (torch.Tensor): lengths of memory buffer
                Dimension: (batch_size,)
            mem_preds (torch.Tensor): speaker predictions for memory buffer
                Dimension: (batch_size, mem_len, num_spk)
            fifo (torch.Tensor): FIFO queue to save the embeddings from the latest chunks.
                Dimension: (batch_size, fifo_len, emb_dim)
            fifo_lengths (torch.Tensor): lengths of FIFO queue
                Dimension: (batch_size,)
            chunk (torch.Tensor): chunk of embeddings to be predicted
                Dimension: (batch_size, lc+chunk_len+rc, emb_dim)
            chunk_lengths (torch.Tensor): lengths of current chunk
                Dimension: (batch_size,)
            preds (torch.Tensor): speaker predictions of the [mem + fifo + chunk] embeddings
                Dimension: (batch_size, mem_len + fifo_len + lc+chunk_len+rc, num_spks)
            lc and rc (int): left & right offset of the chunk,
                only the chunk[:, lc:chunk_len+lc] is used for FIFO queue and memory update

        Returns:
            mem (torch.Tensor): updated memory buffer
                Dimension: (batch_size, mem_len, emb_dim)
            mem_lengths (torch.Tensor): unpated lengths of memory buffer
                Dimension: (batch_size,)
            fifo (torch.Tensor): updated FIFO queue
                Dimension: (batch_size, fifo_len, emb_dim)
            fifo_lengths (torch.Tensor): updated lengths of FIFO queue
                Dimension: (batch_size,)
            mem_preds (torch.Tensor): updated speaker predictions for memory buffer
                Dimension: (batch_size, mem_len, num_spk)
            fifo_preds (torch.Tensor): speaker predictions for FIFO queuer
                Dimension: (batch_size, fifo_len, num_spk)
            chunk_preds (torch.Tensor): speaker predictions of the chunk embeddings
                Dimension: (batch_size, chunk_len, num_spks)
        """

        batch_size, _, emb_dim = mem.shape
        n_spk = preds.shape[2]

        max_mem_len, max_fifo_len, max_chunk_len = mem.shape[1], fifo.shape[1], chunk.shape[1] - lc - rc

        if self.fifo_len == 0:
            max_pop_out_len = max_chunk_len
        elif self.mem_refresh_rate == 0:
            max_pop_out_len = self.fifo_len
        else:
            max_pop_out_len = min(self.mem_refresh_rate * self.step_len, self.fifo_len)

        fifo_preds = torch.zeros((batch_size, max_fifo_len, n_spk), device=preds.device)
        chunk_preds = torch.zeros((batch_size, max_chunk_len, n_spk), device=preds.device)
        chunk_lengths = (chunk_lengths - lc).clamp(min=0,max=max_chunk_len)
        updated_fifo = torch.zeros((batch_size, max_fifo_len + max_chunk_len, emb_dim), device=preds.device)
        updated_fifo_preds = torch.zeros((batch_size, max_fifo_len + max_chunk_len, n_spk), device=preds.device)
        updated_mem = torch.zeros((batch_size, max_mem_len + max_pop_out_len, emb_dim), device=preds.device)
        updated_mem_preds = torch.full((batch_size, max_mem_len + max_pop_out_len, n_spk), -0.1, device=preds.device)

        for batch_index in range(batch_size):
            mem_len = mem_lengths[batch_index].item()
            fifo_len = fifo_lengths[batch_index].item()
            chunk_len = chunk_lengths[batch_index].item()
            fifo_preds[batch_index, :fifo_len, :] = preds[batch_index, mem_len:mem_len+fifo_len, :]
            chunk_preds[batch_index, :chunk_len, :] = preds[batch_index, mem_len+fifo_len+lc:mem_len+fifo_len+lc+chunk_len]
            updated_mem[batch_index, :mem_len, :] = mem[batch_index, :mem_len, :]
            updated_mem_preds[batch_index, :mem_len, :] = mem_preds[batch_index, :mem_len, :]
            updated_fifo[batch_index, :fifo_len, :] = fifo[batch_index, :fifo_len, :]
            updated_fifo_preds[batch_index, :fifo_len, :] = fifo_preds[batch_index, :fifo_len, :]

            # append chunk to fifo
            fifo_lengths[batch_index] += chunk_len
            updated_fifo[batch_index, fifo_len:fifo_len+chunk_len, :] = chunk[batch_index, lc:lc+chunk_len, :]
            updated_fifo_preds[batch_index, fifo_len:fifo_len+chunk_len, :] = chunk_preds[batch_index, :chunk_len, :]
            if fifo_len + chunk_len > max_fifo_len:
                # move pop_out_len first frames of fifo to memory
                pop_out_len = min(max_pop_out_len, fifo_len + chunk_len)
                mem_lengths[batch_index] += pop_out_len
                updated_mem[batch_index, mem_len:mem_len+pop_out_len, :] = updated_fifo[batch_index, :pop_out_len, :]
                if updated_mem_preds[batch_index, 0, 0] >= 0:
                    # memory already compressed at least once
                    updated_mem_preds[batch_index, mem_len:mem_len+pop_out_len, :] = updated_fifo_preds[batch_index, :pop_out_len, :]
                elif mem_len + pop_out_len > self.mem_len:
                    # will compress memory for the first time
                    updated_mem_preds[batch_index, :mem_len, :] = preds[batch_index, :mem_len, :]
                    updated_mem_preds[batch_index, mem_len:mem_len+pop_out_len, :] = updated_fifo_preds[batch_index, :pop_out_len, :]
                fifo_lengths[batch_index] -= pop_out_len
                new_fifo_len = fifo_lengths[batch_index].item()
                updated_fifo[batch_index, :new_fifo_len, :] = updated_fifo[batch_index, pop_out_len:pop_out_len+new_fifo_len, :]
                updated_fifo[batch_index, new_fifo_len:, :] = 0

        fifo = updated_fifo[:, :max_fifo_len, :]

        # update memory
        need_compress = (mem_lengths > self.mem_len)
        mem = updated_mem[:, :self.mem_len:, :]
        mem_preds = updated_mem_preds[:, :self.mem_len:, :]

        idx = torch.where(need_compress)[0]
        if len(idx) > 0:
            mem[idx], mem_preds[idx], _ = self._compress_spk_cache(emb_seq=updated_mem[idx], preds=updated_mem_preds[idx], permute_spk=False)
            mem_lengths[idx] = mem_lengths[idx].clamp(max=self.mem_len)

        if self.log:
            logging.info(f"MC mem: {mem.shape}, chunk: {chunk.shape}, fifo: {fifo.shape}, chunk_preds: {chunk_preds.shape}")

        return mem, mem_lengths, fifo, fifo_lengths, mem_preds, fifo_preds, chunk_preds

    def update_memory_fifo(
        self,
        mem,
        mem_preds,
        fifo,
        chunk,
        preds,
        spk_perm: Optional[torch.Tensor],
        lc: int = 0,
        rc: int = 0
    ):
        """
        update the FIFO queue and memory buffer with the chunk of embeddings and speaker predictions
        Args:
            mem (torch.Tensor): memory buffer to save the embeddings from start
                Dimension: (batch_size, mem_len, emb_dim)
            fifo (torch.Tensor): FIFO queue to save the embeddings from the latest chunks 
                Dimension: (batch_size, fifo_len, emb_dim)
            chunk (torch.Tensor): chunk of embeddings to be predicted
                Dimension: (batch_size, lc+chunk_len+rc, emb_dim)
            preds (torch.Tensor): speaker predictions of the [mem + fifo + chunk] embeddings
                Dimension: (batch_size, mem_len + fifo_len + lc+chunk_len+rc, num_spks)
            lc and rc (int): left & right offset of the chunk,
                only the chunk[:, lc:chunk_len+lc] is used for FIFO queue and memory update

        Returns:
            mem (torch.Tensor): updated memory buffer
                Dimension: (batch_size, mem_len, emb_dim)
            fifo (torch.Tensor): updated FIFO queue
                Dimension: (batch_size, fifo_len, emb_dim)
            chunk_preds (torch.Tensor): speaker predictions of the chunk embeddings
                Dimension: (batch_size, chunk_len, num_spks)
        """

        batch_size, _, emb_dim = mem.shape

        mem_len, fifo_len, chunk_len = mem.shape[1], fifo.shape[1], chunk.shape[1] - lc - rc
        if spk_perm is not None:
            inv_spk_perm = torch.stack([torch.argsort(spk_perm[batch_index]) for batch_index in range(batch_size)])
            preds = torch.stack([preds[batch_index, :, inv_spk_perm[batch_index]] for batch_index in range(batch_size)])

        fifo_preds = preds[:, mem_len:mem_len + fifo_len]
        chunk = chunk[:, lc:chunk_len + lc]
        chunk_preds = preds[:, mem_len + fifo_len + lc:mem_len + fifo_len + chunk_len + lc]

        if self.fifo_len == 0:
            if mem_len == 0 and self.init_step_len < self.step_len:
                fifo = torch.cat([fifo, chunk], dim=1)
                if fifo_len >= self.step_len:
                    pop_out_embs = fifo
                    pop_out_preds = torch.cat([fifo_preds, chunk_preds], dim=1)
                    fifo = self.init_memory(batch_size, emb_dim, mem.device)
                else:
                    pop_out_embs, pop_out_preds = self.init_memory(batch_size, emb_dim, mem.device), self.init_memory(batch_size, self.unit_n_spks, mem.device)
            else:
                assert fifo_len == self.fifo_len
                pop_out_embs, pop_out_preds = chunk, chunk_preds
        else:
            fifo = torch.cat([fifo, chunk], dim=1)
            if fifo.size(1) <= self.fifo_len:
                pop_out_embs, pop_out_preds = self.init_memory(batch_size, emb_dim, mem.device), self.init_memory(batch_size, self.unit_n_spks, mem.device)
            else:
                if self.mem_refresh_rate == 0: # clear fifo queue when it reaches the max_fifo_len and update memory buffer
                    pop_out_embs  = fifo[:, :fifo_len]
                    pop_out_preds = fifo_preds
                    fifo = self.init_memory(batch_size, emb_dim, mem.device)
                elif self.mem_refresh_rate == 1: # pop out the oldest chunk from the fifo queue and update memory buffer
                    pop_out_embs  = fifo[:, :-self.fifo_len]
                    pop_out_preds = fifo_preds[:, :pop_out_embs.shape[1]]
                    fifo = fifo[:, -self.fifo_len:]
                    assert pop_out_embs.shape[1] > 0
                else:
                    # pop out self.mem_refresh_rate oldest chunks from the fifo queue and update memory buffer
                    pop_out_embs = fifo[:, :chunk_len*self.mem_refresh_rate]
                    pop_out_preds = fifo_preds[:, :pop_out_embs.shape[1]]
                    fifo = fifo[:, pop_out_embs.shape[1]:]

        if pop_out_embs.shape[1] > 0: # only update memory buffer when pop_out_embs is not empty
            mem = torch.cat([mem, pop_out_embs], dim=1)
            if mem_preds is not None: # if memory has been already updated at least once
                mem_preds = torch.cat([mem_preds, pop_out_preds], dim=1)
            if mem.shape[1] > self.mem_len:
                if mem_preds is None: # if this is a first memory update
                    mem_preds = torch.cat([preds[:, :mem_len], pop_out_preds], dim=1)
                mem, mem_preds, spk_perm = self._compress_spk_cache(emb_seq=mem, preds=mem_preds, permute_spk=self.training)

        if self.log:
            logging.info(f"MC mem: {mem.shape}, chunk: {chunk.shape}, fifo: {fifo.shape}, chunk_preds: {chunk_preds.shape}")

        return mem, fifo, mem_preds, fifo_preds, chunk_preds, spk_perm

    def _boost_topk_scores(
        self,
        scores,
        n_boost_per_spk: int,
        batch_size: int,
        n_spk: int,
        scale_factor: float = 1.0,
        offset: float = 0.5
    ) -> torch.Tensor:
        """
        Increases `n_boost_per_spk` highest scores for each speaker.

        Args:
            scores (torch.Tensor): Tensor containing scores for each frame and speaker.
                Shape: (batch_size, n_frames, n_spk)
            n_boost_per_spk (int): Number of frames to boost per speaker.
            batch_size (int): Number of samples in a batch.
            n_spk (int): Number of speakers.
            scale_factor (float): Scaling factor for boosting scores. Defaults to 1.0.
            offset (float): Offset for score adjustment. Defaults to 0.5.

        Returns:
            scores (torch.Tensor): Tensor containing scores for each frame and speaker after boosting.
        """
        _, topk_indices = torch.topk(scores, n_boost_per_spk, dim=1, largest=True, sorted=False) # Shape: (batch_size, n_boost_per_spk, n_spk)
        batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1)
        speaker_indices = torch.arange(n_spk).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_spk)
        # Boost scores corresponding to topk_indices; but scores for disabled frames will remain -inf
        scores[batch_indices, topk_indices, speaker_indices] -= scale_factor*math.log(offset)
        return scores

    def _get_silence_profile(self, emb_seq, preds):
        """
        Condition for frame being silence
        Get mean silence embedding tensor
        Get frame importance scores
        """
        is_sil = (preds.sum(dim=2) < self.sil_threshold) * (preds.sum(dim=2) > -0.1) # Shape: (batch_size, n_frames)
        is_sil = is_sil.unsqueeze(-1) # Shape: (batch_size, n_frames, 1)
        emb_seq_sil = torch.where(is_sil, emb_seq, torch.tensor(0.0)) # Shape: (batch_size, n_frames, emb_dim)
        emb_seq_sil_sum = emb_seq_sil.sum(dim=1) # Shape: (batch_size, emb_dim)
        sil_count = is_sil.sum(dim=1).clamp(min=1) # Shape: (batch_size)
        emb_seq_sil_mean = emb_seq_sil_sum / sil_count # Shape: (batch_size, emb_dim)
        emb_seq_sil_mean = emb_seq_sil_mean.unsqueeze(1).expand(-1, self.mem_len, -1) # Shape: (batch_size, mem_len, emb_dim)
        return emb_seq_sil_mean
    
    def _get_log_pred_scores(self, preds, n_spk):
        log_probs = torch.log(torch.clamp(preds, min=self.pred_score_threshold))
        log_1_probs = torch.log(torch.clamp(1.0-preds, min=self.pred_score_threshold))
        log_1_probs_sum = log_1_probs.sum(dim=2).unsqueeze(-1).expand(-1, -1, n_spk)
        scores = log_probs - log_1_probs + log_1_probs_sum - math.log(0.5)
        return scores

    def _topk_operations(self, scores, preds, emb_seq):
        batch_size, n_frames, n_spk = preds.shape
        emb_dim = emb_seq.shape[2]

        # Concatenate scores for all speakers
        # Get mem_len frames with highest scores
        # Replace topk_indices corresponding to -inf score with a placeholder index self.max_index
        # Sort topk_indices to preserve original order of frames
        # Get correct indices corresponding to original frames
        scores_flatten = scores.permute(0,2,1).reshape(batch_size, -1)
        topk_values, topk_indices = torch.topk(scores_flatten, self.mem_len, dim=1, largest=True, sorted=False) 
        valid_topk_mask = (topk_values != float('-inf'))
        topk_indices = torch.where(valid_topk_mask, topk_indices, torch.tensor(self.max_index))
        topk_indices_sorted, _ = torch.sort(topk_indices, dim=1) # Shape: (batch_size, mem_len)
        is_inf = (topk_indices_sorted == self.max_index)
        topk_indices_sorted = torch.remainder(topk_indices_sorted, n_frames + self.mem_sil_frames_per_spk)
        is_inf += (topk_indices_sorted >= n_frames)
        topk_indices_sorted[is_inf] = 0 # Set a placeholder index to make gather work

        # Expand topk_indices_sorted to emb_dim in last dimension to use gather
        # Gather memory buffer including placeholder embeddings for silence frames
        # Get mean silence embedding
        # Replace the placeholder embeddings with actual mean silence embedding
        topk_indices_expanded = topk_indices_sorted.unsqueeze(-1).expand(-1, -1, emb_dim) 
        emb_seq_gathered = torch.gather(emb_seq, 1, topk_indices_expanded) # Shape: (batch_size, mem_len, emb_dim)
        emb_seq_sil_mean = self._get_silence_profile(emb_seq, preds)
        memory_buff = torch.where(is_inf.unsqueeze(-1), emb_seq_sil_mean, emb_seq_gathered)

        # Expand topk indices to n_spk in last dimension to use gather
        # Gather memory preds including placeholder preds for silence frames
        # Replace the placeholder preds with zeros
        topk_indices_expanded_spk = topk_indices_sorted.unsqueeze(-1).expand(-1, -1, n_spk)
        preds_gathered = torch.gather(preds, 1, topk_indices_expanded_spk) # Shape: (batch_size, mem_len, n_spk)
        mem_preds = torch.where(is_inf.unsqueeze(-1), torch.tensor(0.0), preds_gathered)

        return memory_buff, mem_preds

    def _get_max_perm_index(self, scores):
        batch_size, _, n_spk = scores.shape
        is_high = scores > 0 # high score usually means that only current speaker is speaking
        zero_indices = torch.where(is_high.sum(dim=1) == 0)
        max_perm_index = torch.full((batch_size,), n_spk, dtype=torch.long, device=scores.device)
        max_perm_index.scatter_reduce_(0, zero_indices[0], zero_indices[1], reduce="amin", include_self=False)
        return max_perm_index

    def _disable_low_scores(self, preds, scores, min_high_scores_per_spk):
        batch_size, _, n_spk = scores.shape
        is_speech = preds > 0.5
        # Replace scores for non-speech with -inf
        scores = torch.where(is_speech, scores, torch.tensor(float('-inf')))

        is_high = scores > 0 # high score usually means that only current speaker is speaking
        # Replace low scores (usually overlapped speech) with -inf
        # This will be applied only if a speaker has at least min_high_scores_per_spk high-scored frames
        is_low_replace = (~is_high) * is_speech * (is_high.sum(dim=1).unsqueeze(1) >= min_high_scores_per_spk)
        scores = torch.where(is_low_replace, torch.tensor(float('-inf')), scores)
        return scores

    def _compress_spk_cache(self, emb_seq, preds, permute_spk: bool=False):
        """
        Compresses speaker cache for streaming inference
        Keeps mem_len most important frames out of input n_frames, based on preds

        Args:
            emb_seq (torch.Tensor): Tensor containing n_frames > mem_len (mem+chunk) embeddings
                Dimension: (batch_size, n_frames, emb_dim)
            preds (torch.Tensor): Tensor containing n_frames > mem_len (mem+chunk) speaker sigmoid outputs
                Dimension: (batch_size, n_frames, n_spk)
            permute_spk (bool): if true, will generate a random permutation of existing speakers

        Returns:
            memory_buff (torch.Tensor): concatenation of num_spk subtensors of emb_seq corresponding to each speaker
            each of subtensors contains (mem_len//num_spk) frames out of n_frames
                Dimension: (batch_size, mem_len, emb_dim)
            mem_preds (torch.Tensor): predictions for memory buffer
                Dimension: (batch_size, mem_len, n_spk)
            spk_perm (torch.Tensor): random speaker permutation tensor, or None
                Dimension: (batch_size, n_spk)
        """
        batch_size, n_frames, n_spk = preds.shape
        emb_dim = emb_seq.shape[2]
        mem_len_per_spk = self.mem_len // n_spk - self.mem_sil_frames_per_spk
        strong_boost_per_spk = math.floor(mem_len_per_spk * self.strong_boost_rate)
        weak_boost_per_spk = math.floor(mem_len_per_spk  * self.weak_boost_rate)
        min_high_scores_per_spk = math.floor(mem_len_per_spk * self.min_high_scores_rate)

        scores = self._get_log_pred_scores(preds, n_spk)
        scores = self._disable_low_scores(preds, scores, min_high_scores_per_spk)
        max_perm_index = self._get_max_perm_index(scores)

        # Boost newly added frames
        if self.scores_boost_latest > 0:
            scores[:,self.mem_len:,:] += self.scores_boost_latest

        if self.training:
            # Add random noise to scores
            if self.scores_add_rnd > 0:
                scores += torch.rand(batch_size, n_frames, n_spk, device=scores.device) * self.scores_add_rnd

        if permute_spk:
            # Generate a random permutation of speakers
            spk_perm_list = []
            for batch_index in range(batch_size):
                rand_perm_inds = torch.randperm(max_perm_index[batch_index].item())
                linear_inds = torch.arange(max_perm_index[batch_index].item(), n_spk)
                spk_perm_list.append(torch.cat([rand_perm_inds, linear_inds]))
            spk_perm = torch.stack(spk_perm_list).to(preds.device)
            scores = torch.stack([scores[batch_index, :, spk_perm[batch_index]] for batch_index in range(batch_size)])
        else:
            spk_perm = None

        # Strong boosting to ensure each speaker has at least K frames in speaker cache
        scores = self._boost_topk_scores(scores, strong_boost_per_spk, batch_size, n_spk, scale_factor=2)
        # Weak boosting to prevent dominance of one speaker in speaker cache
        scores = self._boost_topk_scores(scores, weak_boost_per_spk, batch_size, n_spk, scale_factor=1)

        if self.mem_sil_frames_per_spk > 0: # Add number of silence frames in the end of each block
            scores = torch.cat([scores, torch.full((batch_size, self.mem_sil_frames_per_spk, n_spk), self.max_score, device=scores.device)], dim=1) # Shape: (batch_size, n_frames + mem_sil_frames_per_spk, n_spk)

        memory_buff, mem_preds = self._topk_operations(scores, preds, emb_seq)
        return memory_buff, mem_preds, spk_perm
