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


import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.utils import logging
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from typing import List, Tuple

__all__ = ['SortformerModules']


def concat_and_pad(embs: List[torch.Tensor], lengths: List[torch.Tensor]):
    """Concatenates lengths[i] first embeddings of embs[i], and pads the rest elements with zeros.
    Args:
        embs: List of embeddings Tensors of (B, T_i, D) shape
        lengths: List of lengths Tensors of (B,) shape

    Returns:
        output: concatenated embeddings Tensor of (B, T, D) shape
        total_lengths: output lengths Tensor of (B,) shape
    """

    assert len(embs) == len(lengths)
    device = embs[0].device
    dtype = embs[0].dtype
    B, D = embs[0].shape[0], embs[0].shape[2]

    total_lengths = torch.sum(torch.stack(lengths), dim=0)
    max_length = total_lengths.max().item()

    output = torch.zeros(B, max_length, D, device=device, dtype=dtype)
    start_indices = torch.zeros(B, dtype=torch.int64, device=device)

    for E, L in zip(embs, lengths):
        end_indices = start_indices + L
        for b in range(B):
            output[b, start_indices[b]:end_indices[b]] = E[b, :L[b]]
        start_indices = end_indices

    return output, total_lengths


class SortformerModules(NeuralModule, Exportable):
    """
    A class including auxiliary functions for Sortformer models.
    This class contains and will contain the following functions that performs streaming features,
    and any neural layers that are not included in the NeMo neural modules (e.g. Transformer, Fast-Conformer).
    """

    def init_weights(self, m):
        """Init weights for linear layers."""
        if isinstance(m, nn.Linear):
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
        scores_boost_latest: float = 0.05,
        sil_threshold: float = 0.2,
        strong_boost_rate: float = 0.75,
        weak_boost_rate: float = 1.5,
        min_pos_scores_rate: float = 0.5,
    ):
        super().__init__()
        self.spkcache_sil_frames_per_spk = mem_sil_frames_per_spk
        self.step_left_context = step_left_context
        self.step_right_context = step_right_context
        self.subsampling_factor = subsampling_factor
        self.spkcache_len = mem_len
        self.fifo_len = fifo_len
        self.step_len = step_len
        self.spkcache_refresh_rate = mem_refresh_rate
        self.fc_d_model = fc_d_model
        self.tf_d_model = tf_d_model
        self.hidden_size = tf_d_model
        self.n_spk: int = num_spks
        self.hidden_to_spks = nn.Linear(2 * self.hidden_size, self.n_spk)
        self.first_hidden_to_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.single_hidden_to_spks = nn.Linear(self.hidden_size, self.n_spk)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_proj = nn.Linear(self.fc_d_model, self.tf_d_model)
        self.log = False
        self.causal_attn_rate = causal_attn_rate
        self.causal_attn_rc = causal_attn_rc
        self.use_causal_eval = use_causal_eval
        self.scores_add_rnd = scores_add_rnd
        self.init_step_len = init_step_len
        self.max_index = max_index
        self.pred_score_threshold = pred_score_threshold
        self.scores_boost_latest = scores_boost_latest
        self.sil_threshold = sil_threshold
        self.strong_boost_rate = strong_boost_rate
        self.weak_boost_rate = weak_boost_rate
        self.min_pos_scores_rate = min_pos_scores_rate
        self.visualization = False

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

    def streaming_feat_loader(
        self,
        feat_seq,
        feat_seq_length,
        feat_seq_offset
    ) -> Tuple[int, torch.Tensor, torch.Tensor, int, int]:
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
            logging.info(
                f"feat_len={feat_len}, num_chunks={num_chunks}, "
                f"feat_seq_length={feat_seq_length}, feat_seq_offset={feat_seq_offset}"
            )
        stt_feat, end_feat, step_idx = 0, 0, 0
        current_step_len = min(self.init_step_len, self.step_len)
        while end_feat < feat_len:
            left_offset = min(self.step_left_context * self.subsampling_factor, stt_feat)
            end_feat = min(stt_feat + current_step_len * self.subsampling_factor, feat_len)
            right_offset = min(self.step_right_context * self.subsampling_factor, feat_len - end_feat)
            chunk_feat_seq = feat_seq[:, :, stt_feat - left_offset:end_feat + right_offset]
            feat_lengths = (
                (feat_seq_length + feat_seq_offset - stt_feat + left_offset)
                .clamp(0, chunk_feat_seq.shape[2])
            )
            feat_lengths = feat_lengths * (feat_seq_offset < end_feat)
            stt_feat = end_feat
            chunk_feat_seq_t = torch.transpose(chunk_feat_seq, 1, 2)
            if self.log:
                logging.info(
                    f"step_idx: {step_idx}, current step len: {current_step_len}, "
                    f"chunk_feat_seq_t shape: {chunk_feat_seq_t.shape}, "
                    f"chunk_feat_lengths: {feat_lengths}"
                )
            yield step_idx, chunk_feat_seq_t, feat_lengths, left_offset, right_offset
            step_idx += 1
            if end_feat >= self.step_len * self.subsampling_factor:
                current_step_len = self.step_len
            elif (
                end_feat < self.step_len * self.subsampling_factor
                and current_step_len < self.step_len
                and end_feat >= 4 * current_step_len * self.subsampling_factor
            ):
                current_step_len *= 2

    def forward_speaker_sigmoids(self, hidden_out):
        """
        The final layer that renders speaker probabilities in Sigmoid activation function.

        Args:
            hidden_out (torch.Tensor): tensor containing hidden states from the encoder
                Dimension: (batch_size, n_frames, hidden_dim)

        Returns:
            preds (torch.Tensor): tensor containing speaker probabilities in Sigmoid activation function
                Dimension: (batch_size, n_frames, n_spk)
        """
        hidden_out = self.dropout(F.relu(hidden_out))
        hidden_out = self.first_hidden_to_hidden(hidden_out)
        hidden_out = self.dropout(F.relu(hidden_out))
        spk_preds = self.single_hidden_to_spks(hidden_out)
        preds = nn.Sigmoid()(spk_preds)
        return preds

    def concat_embs(
        self,
        list_of_tensors=List[torch.Tensor],
        return_lengths: bool = False,
        dim: int = 1,
        device: torch.device = None
    ):
        """
        Concatenate a list of tensors along the specified dimension.

        Args:
            list_of_tensors (List[torch.Tensor]): List of tensors to concatenate
            return_lengths (bool): Whether to return lengths of the concatenated tensors
            dim (int): Concatenation axis
            device (torch.device): device to use for tensor operations

        Returns:
            embs (torch.Tensor): concatenated tensor
        """
        embs = torch.cat(list_of_tensors, dim=dim).to(device)
        lengths = torch.tensor(embs.shape[1]).repeat(embs.shape[0]).to(device)
        if return_lengths:
            return embs, lengths
        else:
            return embs

    def streaming_update_async(
        self,
        streaming_state,
        chunk,
        chunk_lengths,
        preds,
        lc: int = 0,
        rc: int = 0
    ):
        """
        Update the speaker cache and FIFO queue with the chunk of embeddings and speaker predictions.
        Asynchronous version, which means speaker cache, FIFO and chunk may have different lengths within a batch.
        Should be used for real stremaing applicaitons.

        Args:
            spkcache (torch.Tensor): speaker cache to save the embeddings from start
                Dimension: (batch_size, spkcache_len, emb_dim)
            spkcache_lengths (torch.Tensor): lengths of speaker cache
                Dimension: (batch_size,)
            spkcache_preds (torch.Tensor): speaker activity probabilities for speaker cache
                Dimension: (batch_size, spkcache_len, n_spk)
            fifo (torch.Tensor): FIFO queue to save the embeddings from the latest chunks.
                Dimension: (batch_size, fifo_len, emb_dim)
            fifo_lengths (torch.Tensor): lengths of FIFO queue
                Dimension: (batch_size,)
            chunk (torch.Tensor): chunk of embeddings to be predicted
                Dimension: (batch_size, lc+chunk_len+rc, emb_dim)
            chunk_lengths (torch.Tensor): lengths of current chunk
                Dimension: (batch_size,)
            preds (torch.Tensor): speaker predictions of the [spkcache + fifo + chunk] embeddings
                Dimension: (batch_size, spkcache_len + fifo_len + lc+chunk_len+rc, num_spks)
            lc and rc (int): left & right offset of the chunk,
                only the chunk[:, lc:chunk_len+lc] is used for update of speaker cache and FIFO queue

        Returns:
            spkcache (torch.Tensor): updated speaker cache
                Dimension: (batch_size, spkcache_len, emb_dim)
            spkcache_lengths (torch.Tensor): updated lengths of speaker cache
                Dimension: (batch_size,)
            fifo (torch.Tensor): updated FIFO queue
                Dimension: (batch_size, fifo_len, emb_dim)
            fifo_lengths (torch.Tensor): updated lengths of FIFO queue
                Dimension: (batch_size,)
            spkcache_preds (torch.Tensor): updated speaker predictions for speaker cache
                Dimension: (batch_size, spkcache_len, num_spk)
            fifo_preds (torch.Tensor): speaker predictions for FIFO queuer
                Dimension: (batch_size, fifo_len, num_spk)
            chunk_preds (torch.Tensor): speaker predictions of the chunk embeddings
                Dimension: (batch_size, chunk_len, num_spks)
        """
        batch_size, _, emb_dim = chunk.shape
        n_spk = preds.shape[2]

        max_spkcache_len, max_fifo_len, max_chunk_len = streaming_state.spkcache.shape[
            1], streaming_state.fifo.shape[1], chunk.shape[1] - lc - rc

        if self.fifo_len == 0:
            max_pop_out_len = max_chunk_len
        elif self.spkcache_refresh_rate == 0:
            max_pop_out_len = self.fifo_len
        else:
            max_pop_out_len = min(self.spkcache_refresh_rate * self.step_len, self.fifo_len)

        streaming_state.fifo_preds = torch.zeros((batch_size, max_fifo_len, n_spk), device=preds.device)
        chunk_preds = torch.zeros((batch_size, max_chunk_len, n_spk), device=preds.device)
        chunk_lengths = (chunk_lengths - lc).clamp(min=0, max=max_chunk_len)
        updated_fifo = torch.zeros((batch_size, max_fifo_len + max_chunk_len, emb_dim), device=preds.device)
        updated_fifo_preds = torch.zeros((batch_size, max_fifo_len + max_chunk_len, n_spk), device=preds.device)
        updated_spkcache = torch.zeros((batch_size, max_spkcache_len + max_pop_out_len, emb_dim), device=preds.device)
        updated_spkcache_preds = torch.full(
            (batch_size, max_spkcache_len + max_pop_out_len, n_spk), 0.0, device=preds.device)

        for batch_index in range(batch_size):
            spkcache_len = streaming_state.spkcache_lengths[batch_index].item()
            fifo_len = streaming_state.fifo_lengths[batch_index].item()
            chunk_len = chunk_lengths[batch_index].item()
            streaming_state.fifo_preds[batch_index, :fifo_len,
                                       :] = preds[batch_index, spkcache_len:spkcache_len + fifo_len, :]
            chunk_preds[batch_index, :chunk_len, :] = preds[
                batch_index,
                spkcache_len + fifo_len + lc:spkcache_len + fifo_len + lc + chunk_len
            ]
            updated_spkcache[batch_index, :spkcache_len, :] = streaming_state.spkcache[batch_index, :spkcache_len, :]
            updated_spkcache_preds[batch_index, :spkcache_len,
                                   :] = streaming_state.spkcache_preds[batch_index, :spkcache_len, :]
            updated_fifo[batch_index, :fifo_len, :] = streaming_state.fifo[batch_index, :fifo_len, :]
            updated_fifo_preds[batch_index, :fifo_len, :] = streaming_state.fifo_preds[batch_index, :fifo_len, :]

            # append chunk to fifo
            streaming_state.fifo_lengths[batch_index] += chunk_len
            updated_fifo[batch_index, fifo_len:fifo_len + chunk_len, :] = chunk[batch_index, lc:lc + chunk_len, :]
            updated_fifo_preds[batch_index, fifo_len:fifo_len + chunk_len, :] = chunk_preds[batch_index, :chunk_len, :]
            if fifo_len + chunk_len > max_fifo_len:
                # move pop_out_len first frames of FIFO queue to speaker cache
                pop_out_len = min(max_pop_out_len, fifo_len + chunk_len)
                streaming_state.spkcache_lengths[batch_index] += pop_out_len
                updated_spkcache[batch_index, spkcache_len:spkcache_len +
                                 pop_out_len, :] = updated_fifo[batch_index, :pop_out_len, :]
                if updated_spkcache_preds[batch_index, 0, 0] >= 0:
                    # speaker cache already compressed at least once
                    updated_spkcache_preds[batch_index, spkcache_len:spkcache_len +
                                           pop_out_len, :] = updated_fifo_preds[batch_index, :pop_out_len, :]
                elif spkcache_len + pop_out_len > self.spkcache_len:
                    # will compress speaker cache for the first time
                    updated_spkcache_preds[batch_index, :spkcache_len, :] = preds[batch_index, :spkcache_len, :]
                    updated_spkcache_preds[batch_index, spkcache_len:spkcache_len +
                                           pop_out_len, :] = updated_fifo_preds[batch_index, :pop_out_len, :]
                streaming_state.fifo_lengths[batch_index] -= pop_out_len
                new_fifo_len = streaming_state.fifo_lengths[batch_index].item()
                updated_fifo[batch_index, :new_fifo_len, :] = updated_fifo[
                    batch_index, pop_out_len:pop_out_len + new_fifo_len, :
                ].clone()
                updated_fifo[batch_index, new_fifo_len:, :] = 0

        streaming_state.fifo = updated_fifo[:, :max_fifo_len, :]

        # update speaker cache
        need_compress = (streaming_state.spkcache_lengths > self.spkcache_len)
        streaming_state.spkcache = updated_spkcache[:, :self.spkcache_len:, :]
        streaming_state.spkcache_preds = updated_spkcache_preds[:, :self.spkcache_len:, :]

        idx = torch.where(need_compress)[0]
        if len(idx) > 0:
            streaming_state.spkcache[idx], streaming_state.spkcache_preds[idx], _ = self._compress_spkcache(
                emb_seq=updated_spkcache[idx],
                preds=updated_spkcache_preds[idx],
                permute_spk=False
            )
            streaming_state.spkcache_lengths[idx] = streaming_state.spkcache_lengths[
                idx
            ].clamp(max=self.spkcache_len)

        if self.log:
            logging.info(
                f"MC spkcache: {streaming_state.spkcache.shape}, "
                f"chunk: {chunk.shape}, fifo: {streaming_state.fifo.shape}, "
                f"chunk_preds: {chunk_preds.shape}"
            )

        return streaming_state, chunk_preds

    def streaming_update(
        self,
        streaming_state,
        chunk,
        preds,
        lc: int = 0,
        rc: int = 0
    ):
        """
        Update the speaker cache and FIFO queue with the chunk of embeddings and speaker predictions.
        Synchronous version, which means speaker cahce, FIFO queue and chunk have same lengths within a batch.
        Should be used for training and evaluation, not for real stremaing applicaitons.

        Args:
            spkcache (torch.Tensor): speaker cache to save the embeddings from start
                Dimension: (batch_size, spkcache_len, emb_dim)
            spkcache_preds (torch.Tensor): speaker activity probabilities for speaker cache
                Dimension: (batch_size, spkcache_len, num_spk)
            fifo (torch.Tensor): FIFO queue to save the embeddings from the latest chunks.
                Dimension: (batch_size, fifo_len, emb_dim)
            chunk (torch.Tensor): chunk of embeddings to be predicted
                Dimension: (batch_size, lc+chunk_len+rc, emb_dim)
            preds (torch.Tensor): speaker predictions of the [spkcache + fifo + chunk] embeddings
                Dimension: (batch_size, spkcache_len + fifo_len + lc+chunk_len+rc, num_spks)
            spk_perm (torch.Tensor): Tensor containing speaker permutation applied to scores on previous step
                Dimension: (batch_size, n_spk)
            lc and rc (int): left & right offset of the chunk,
                only the chunk[:, lc:chunk_len+lc] is used for update of speaker cache and FIFO queue

        Returns:
            spkcache (torch.Tensor): updated speaker cache
                Dimension: (batch_size, spkcache_len, emb_dim)
            fifo (torch.Tensor): updated FIFO queue
                Dimension: (batch_size, fifo_len, emb_dim)
            spkcache_preds (torch.Tensor): updated speaker predictions for speaker cache
                Dimension: (batch_size, spkcache_len, num_spk)
            fifo_preds (torch.Tensor): speaker predictions for FIFO queuer
                Dimension: (batch_size, fifo_len, num_spk)
            chunk_preds (torch.Tensor): speaker predictions of the chunk embeddings
                Dimension: (batch_size, chunk_len, num_spks)
            spk_perm (torch.Tensor): Tensor containing speaker permutation applied to scores on this step
                Dimension: (batch_size, n_spk)
        """

        batch_size, _, emb_dim = chunk.shape

        spkcache_len, fifo_len, chunk_len = streaming_state.spkcache.shape[
            1], streaming_state.fifo.shape[1], chunk.shape[1] - lc - rc
        if streaming_state.spk_perm is not None:
            inv_spk_perm = torch.stack(
                [torch.argsort(streaming_state.spk_perm[batch_index]) for batch_index in range(batch_size)]
            )
            preds = torch.stack(
                [preds[batch_index, :, inv_spk_perm[batch_index]] for batch_index in range(batch_size)]
            )

        streaming_state.fifo_preds = preds[:, spkcache_len:spkcache_len + fifo_len]
        chunk = chunk[:, lc:chunk_len + lc]
        chunk_preds = preds[:, spkcache_len + fifo_len + lc:spkcache_len + fifo_len + chunk_len + lc]

        if self.fifo_len == 0:
            if spkcache_len == 0 and self.init_step_len < self.step_len:
                streaming_state.fifo = torch.cat([streaming_state.fifo, chunk], dim=1)
                if fifo_len >= self.step_len:
                    pop_out_embs = streaming_state.fifo
                    pop_out_preds = torch.cat([streaming_state.fifo_preds, chunk_preds], dim=1)
                    streaming_state.fifo = torch.zeros(batch_size, 0, emb_dim).to(chunk.device)
                else:
                    pop_out_embs = torch.zeros(batch_size, 0, emb_dim).to(chunk.device)
                    pop_out_preds = torch.zeros(batch_size, 0, self.n_spk).to(chunk.device)
            else:
                assert fifo_len == self.fifo_len
                pop_out_embs, pop_out_preds = chunk, chunk_preds
        else:
            streaming_state.fifo = torch.cat([streaming_state.fifo, chunk], dim=1)
            if streaming_state.fifo.size(1) <= self.fifo_len:
                pop_out_embs = torch.zeros(batch_size, 0, emb_dim).to(chunk.device)
                pop_out_preds = torch.zeros(batch_size, 0, self.n_spk).to(chunk.device)
            else:
                if self.spkcache_refresh_rate == 0:
                    # Clear fifo queue when it reaches the max_fifo_len and update speaker cache
                    pop_out_embs = streaming_state.fifo[:, :fifo_len]
                    pop_out_preds = streaming_state.fifo_preds
                    streaming_state.fifo = torch.zeros(batch_size, 0, emb_dim).to(chunk.device)
                elif self.spkcache_refresh_rate == 1:
                    # Pop out the oldest chunk from the fifo queue and update speaker cache
                    pop_out_embs = streaming_state.fifo[:, :-self.fifo_len]
                    pop_out_preds = streaming_state.fifo_preds[:, :pop_out_embs.shape[1]]
                    streaming_state.fifo = streaming_state.fifo[:, -self.fifo_len:]
                    assert pop_out_embs.shape[1] > 0
                else:
                    # Pop out self.spkcache_refresh_rate oldest chunks from the fifo queue and update speaker cache
                    pop_out_embs = streaming_state.fifo[:, :chunk_len * self.spkcache_refresh_rate]
                    pop_out_preds = streaming_state.fifo_preds[:, :pop_out_embs.shape[1]]
                    streaming_state.fifo = streaming_state.fifo[:, pop_out_embs.shape[1]:]

        if pop_out_embs.shape[1] > 0:  # only update speaker cache when pop_out_embs is not empty
            streaming_state.spkcache = torch.cat([streaming_state.spkcache, pop_out_embs], dim=1)
            if streaming_state.spkcache_preds is not None:  # if speaker cache has been already updated at least once
                streaming_state.spkcache_preds = torch.cat([streaming_state.spkcache_preds, pop_out_preds], dim=1)
            if streaming_state.spkcache.shape[1] > self.spkcache_len:
                if streaming_state.spkcache_preds is None:  # if this is a first update of speaker cache
                    streaming_state.spkcache_preds = torch.cat([preds[:, :spkcache_len], pop_out_preds], dim=1)
                streaming_state.spkcache, \
                streaming_state.spkcache_preds, \
                streaming_state.spk_perm = self._compress_spkcache(
                    emb_seq=streaming_state.spkcache,
                    preds=streaming_state.spkcache_preds,
                    permute_spk=self.training
                )

        if self.log:
            logging.info(
                f"spkcache: {streaming_state.spkcache.shape}, "
                f"chunk: {chunk.shape}, fifo: {streaming_state.fifo.shape}, "
                f"chunk_preds: {chunk_preds.shape}"
            )

        return streaming_state, chunk_preds

    def _boost_topk_scores(
        self,
        scores,
        n_boost_per_spk: int,
        scale_factor: float = 1.0,
        offset: float = 0.5
    ) -> torch.Tensor:
        """
        Increase `n_boost_per_spk` highest scores for each speaker.

        Args:
            scores (torch.Tensor): Tensor containing scores for each frame and speaker.
                Shape: (batch_size, n_frames, n_spk)
            n_boost_per_spk (int): Number of frames to boost per speaker.
            scale_factor (float): Scaling factor for boosting scores. Defaults to 1.0.
            offset (float): Offset for score adjustment. Defaults to 0.5.

        Returns:
            scores (torch.Tensor): Tensor containing scores for each frame and speaker after boosting.
                Shape: (batch_size, n_frames, n_spk)
        """
        batch_size, _, n_spk = scores.shape
        _, topk_indices = torch.topk(scores, n_boost_per_spk, dim=1, largest=True, sorted=False)
        batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1)
        speaker_indices = torch.arange(n_spk).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_spk)
        # Boost scores corresponding to topk_indices; but scores for disabled frames will remain '-inf'
        scores[batch_indices, topk_indices, speaker_indices] -= scale_factor * math.log(offset)
        return scores

    def _get_silence_profile(self, emb_seq, preds):
        """
        Get mean silence embedding from emb_seq sequence.
        Embeddings are considered as silence if sum of corresponding preds is lower than self.sil_threshold.

        Args:
            emb_seq (torch.Tensor): Tensor containing sequence of embeddings.
                Shape: (batch_size, n_frames, emb_dim)
            preds (torch.Tensor): Tensor containing speaker activity probabilities.
                Shape: (batch_size, n_frames, n_spk)

        Returns:
            mean_sil_emb (torch.Tensor): Mean silence embedding tensor.
                Shape: (batch_size, emb_dim)
        """
        is_sil = (preds.sum(dim=2) < self.sil_threshold)
        is_sil = is_sil.unsqueeze(-1)
        emb_seq_sil = torch.where(is_sil, emb_seq, torch.tensor(0.0))  # (batch_size, n_frames, emb_dim)
        emb_seq_sil_sum = emb_seq_sil.sum(dim=1)  # (batch_size, emb_dim)
        sil_count = is_sil.sum(dim=1).clamp(min=1)  # (batch_size)
        mean_sil_emb = emb_seq_sil_sum / sil_count  # (batch_size, emb_dim)
        return mean_sil_emb

    def _get_log_pred_scores(self, preds):
        """
        Get per-frame scores for speakers based on their activity probabilities.
        Scores are log-based and designed to be high for confident prediction of non-overlapped speech.

        Args:
            preds (torch.Tensor): Tensor containing speaker activity probabilities.
                Shape: (batch_size, n_frames, n_spk)

        Returns:
            scores (torch.Tensor): Tensor containing speaker scores.
                Shape: (batch_size, n_frames, n_spk)
        """
        log_probs = torch.log(torch.clamp(preds, min=self.pred_score_threshold))
        log_1_probs = torch.log(torch.clamp(1.0 - preds, min=self.pred_score_threshold))
        log_1_probs_sum = log_1_probs.sum(dim=2).unsqueeze(-1).expand(-1, -1, self.n_spk)
        scores = log_probs - log_1_probs + log_1_probs_sum - math.log(0.5)
        return scores

    def _get_topk_indices(self, scores):
        """
        Get indices corresponding to spkcache_len highest scores, and binary mask for frames in topk to be disabled.
        Disabled frames correspond to either '-inf' score or spkcache_sil_frames_per_spk frames of extra silence
        Mean silence embedding will be used for these frames.

        Args:
            scores (torch.Tensor): Tensor containing speaker scores, including for extra silence frames.
                Shape: (batch_size, n_frames, n_spk)

        Returns:
            topk_indices_sorted (torch.Tensor): Tensor containing frame indices of spkcache_len highest scores.
                Shape: (batch_size, spkcache_len)
            is_disabled (torch.Tensor): Tensor containing binary mask for frames in topk to be disabled.
                Shape: (batch_size, spkcache_len)
        """
        batch_size, n_frames, _ = scores.shape
        n_frames_no_sil = n_frames - self.spkcache_sil_frames_per_spk
        # Concatenate scores for all speakers and get spkcache_len frames with highest scores.
        # Replace topk_indices corresponding to '-inf' score with a placeholder index self.max_index.
        scores_flatten = scores.permute(0, 2, 1).reshape(batch_size, -1)
        topk_values, topk_indices = torch.topk(scores_flatten, self.spkcache_len, dim=1, sorted=False)
        valid_topk_mask = (topk_values != float('-inf'))
        topk_indices = torch.where(valid_topk_mask, topk_indices, torch.tensor(self.max_index))
        # Sort topk_indices to preserve the original order of the frames.
        # Get correct indices corresponding to the original frames
        topk_indices_sorted, _ = torch.sort(topk_indices, dim=1)  # Shape: (batch_size, spkcache_len)
        is_disabled = (topk_indices_sorted == self.max_index)
        topk_indices_sorted = torch.remainder(topk_indices_sorted, n_frames)
        is_disabled += (topk_indices_sorted >= n_frames_no_sil)
        topk_indices_sorted[is_disabled] = 0  # Set a placeholder index to make gather work
        return topk_indices_sorted, is_disabled

    def _gather_spkcache_and_preds(self, emb_seq, preds, topk_indices, is_disabled):
        """
        Gather embeddings from emb_seq and speaker activities from preds corresponding to topk_indices.
        For disabled frames, use mean silence embedding and zero probability instead.

        Args:
            emb_seq (torch.Tensor): Tensor containing sequence of embeddings.
                Shape: (batch_size, n_frames, emb_dim)
            preds (torch.Tensor): Tensor containing speaker activity probabilities.
                Shape: (batch_size, n_frames, n_spk)
            topk_indices (torch.Tensor): Tensor containing indices of frames to gather.
                Shape: (batch_size, spkcache_len)
            is_disabled (torch.Tensor): Tensor containing binary mask for disabled frames.
                Shape: (batch_size, spkcache_len)

        Returns:
            emb_seq_gathered (torch.Tensor): Tensor containing gathered embeddings.
                Shape: (batch_size, spkcache_len, emb_dim)
            preds_gathered (torch.Tensor): Tensor containing gathered speaker activities.
                Shape: (batch_size, spkcache_len, n_spk)
        """
        # To use `torch.gather`, expand `topk_indices` along the last dimension to match `emb_dim`.
        # Gather the speaker cache embeddings, including the placeholder embeddings for silence frames.
        # Finally, replace the placeholder embeddings with actual mean silence embedding.
        emb_dim, n_spk = emb_seq.shape[2], preds.shape[2]
        indices_expanded_emb = topk_indices.unsqueeze(-1).expand(-1, -1, emb_dim)
        emb_seq_gathered = torch.gather(emb_seq, 1, indices_expanded_emb)  # (batch_size, spkcache_len, emb_dim)
        mean_sil_emb = self._get_silence_profile(emb_seq, preds)  # Compute mean silence embedding
        mean_sil_emb_expanded = mean_sil_emb.unsqueeze(1).expand(-1, self.spkcache_len, -1)
        emb_seq_gathered = torch.where(is_disabled.unsqueeze(-1), mean_sil_emb_expanded, emb_seq_gathered)

        # To use `torch.gather`, expand `topk_indices` along the last dimension to match `n_spk`.
        # Gather speaker cache predictions `preds`, including the placeholder `preds` for silence frames.
        # Finally, replace the placeholder `preds` with zeros.
        indices_expanded_spk = topk_indices.unsqueeze(-1).expand(-1, -1, n_spk)
        preds_gathered = torch.gather(preds, 1, indices_expanded_spk)  # (batch_size, spkcache_len, n_spk)
        preds_gathered = torch.where(is_disabled.unsqueeze(-1), torch.tensor(0.0), preds_gathered)
        return emb_seq_gathered, preds_gathered

    def _get_max_perm_index(self, scores):
        """
        Get number of first speakers having at least one positive score.
        These speakers will be randomly permuted during _compress_spkcache (training only).

        Args:
            scores (torch.Tensor): Tensor containing speaker scores.
                Shape: (batch_size, n_frames, n_spk)

        Returns:
            max_perm_index (torch.Tensor): Tensor with number of first speakers to permute.
                Shape: (batch_size)
        """

        batch_size, _, n_spk = scores.shape
        is_pos = scores > 0  # positive score usually means that only current speaker is speaking
        zero_indices = torch.where(is_pos.sum(dim=1) == 0)
        max_perm_index = torch.full((batch_size,), n_spk, dtype=torch.long, device=scores.device)
        max_perm_index.scatter_reduce_(0, zero_indices[0], zero_indices[1], reduce="amin", include_self=False)
        return max_perm_index

    def _disable_low_scores(self, preds, scores, min_pos_scores_per_spk: int):
        """
        Sets scores for non-speech to '-inf'.
        Also sets non-positive scores to '-inf', if there are at least min_pos_scores_per_spk positive scores.

        Args:
            preds (torch.Tensor): Tensor containing speaker activity probabilities.
                Shape: (batch_size, n_frames, n_spk)
            scores (torch.Tensor): Tensor containing speaker importance scores.
                Shape: (batch_size, n_frames, n_spk)
            min_pos_scores_per_spk (int): if number of positive scores for a speaker is greater than this,
                then all non-positive scores for this speaker will be disabled, i.e. set to '-inf'.

        Returns:
            scores (torch.Tensor): Tensor containing speaker scores.
                Shape: (batch_size, n_frames, n_spk)
        """
        # Replace scores for non-speech with '-inf'.
        is_speech = preds > 0.5
        scores = torch.where(is_speech, scores, torch.tensor(float('-inf')))

        # Replace non-positive scores (usually overlapped speech) with '-inf'
        # This will be applied only if a speaker has at least min_pos_scores_per_spk positive-scored frames
        is_pos = scores > 0  # positive score usually means that only current speaker is speaking
        is_nonpos_replace = (~is_pos) * is_speech * (is_pos.sum(dim=1).unsqueeze(1) >= min_pos_scores_per_spk)
        scores = torch.where(is_nonpos_replace, torch.tensor(float('-inf')), scores)
        return scores

    def _permute_speakers(self, scores, max_perm_index):
        """
        Create a random permutation of scores max_perm_index first speakers.

        Args:
            scores (torch.Tensor): Tensor containing speaker scores.
                Shape: (batch_size, n_frames, n_spk)
            max_perm_index (torch.Tensor): Tensor with number of first speakers to permute.
                Shape: (batch_size)

        Returns:
            scores (torch.Tensor): Tensor with permuted scores.
                Shape: (batch_size, n_frames, n_spk)
            spk_perm (torch.Tensor): Tensor containing speaker permutation applied to scores.
                Dimension: (batch_size, n_spk)
        """
        spk_perm_list, scores_list = [], []
        batch_size, _, n_spk = scores.shape
        for batch_index in range(batch_size):
            rand_perm_inds = torch.randperm(max_perm_index[batch_index].item())
            linear_inds = torch.arange(max_perm_index[batch_index].item(), n_spk)
            permutation = torch.cat([rand_perm_inds, linear_inds])
            spk_perm_list.append(permutation)
            scores_list.append(scores[batch_index, :, permutation])
        spk_perm = torch.stack(spk_perm_list).to(scores.device)
        scores = torch.stack(scores_list).to(scores.device)
        return scores, spk_perm

    def _compress_spkcache(self, emb_seq, preds, permute_spk: bool = False):
        """
        Compress speaker cache for streaming inference.
        Keep spkcache_len most important frames out of input n_frames, based on preds.

        Args:
            emb_seq (torch.Tensor): Tensor containing n_frames > spkcache_len embeddings
                Dimension: (batch_size, n_frames, emb_dim)
            preds (torch.Tensor): Tensor containing n_frames > spkcache_len speaker activity probabilities
                Dimension: (batch_size, n_frames, n_spk)
            permute_spk (bool): If true, will generate a random permutation of existing speakers

        Returns:
            spkcache (torch.Tensor): Tensor containing spkcache_len most important embeddings from emb_seq.
            Embeddings are ordered by speakers. Within each speaker, original order of frames is kept.
                Dimension: (batch_size, spkcache_len, emb_dim)
            spkcache_preds (torch.Tensor): predictions corresponding to speaker cache
                Dimension: (batch_size, spkcache_len, n_spk)
            spk_perm (torch.Tensor): random speaker permutation tensor if permute_spk=True, otherwise None
                Dimension: (batch_size, n_spk)
        """
        batch_size, n_frames, n_spk = preds.shape
        spkcache_len_per_spk = self.spkcache_len // n_spk - self.spkcache_sil_frames_per_spk
        strong_boost_per_spk = math.floor(spkcache_len_per_spk * self.strong_boost_rate)
        weak_boost_per_spk = math.floor(spkcache_len_per_spk * self.weak_boost_rate)
        min_pos_scores_per_spk = math.floor(spkcache_len_per_spk * self.min_pos_scores_rate)

        scores = self._get_log_pred_scores(preds)
        scores = self._disable_low_scores(preds, scores, min_pos_scores_per_spk)

        if permute_spk:  # Generate a random permutation of speakers
            max_perm_index = self._get_max_perm_index(scores)
            scores, spk_perm = self._permute_speakers(scores, max_perm_index)
        else:
            spk_perm = None

        if self.scores_boost_latest > 0:  # Boost newly added frames
            scores[:, self.spkcache_len:, :] += self.scores_boost_latest

        if self.training:
            if self.scores_add_rnd > 0:  # Add random noise to scores
                scores += torch.rand(batch_size, n_frames, n_spk, device=scores.device) * self.scores_add_rnd

        # Strong boosting to ensure each speaker has at least K frames in speaker cache
        scores = self._boost_topk_scores(scores, strong_boost_per_spk, scale_factor=2)
        # Weak boosting to prevent dominance of one speaker in speaker cache
        scores = self._boost_topk_scores(scores, weak_boost_per_spk, scale_factor=1)

        if self.spkcache_sil_frames_per_spk > 0:  # Add number of silence frames in the end of each block
            pad = torch.full(
                (batch_size, self.spkcache_sil_frames_per_spk, n_spk), float('inf'), device=scores.device
            )
            scores = torch.cat([scores, pad], dim=1)  # (batch_size, n_frames + spkcache_sil_frames_per_spk, n_spk)

        topk_indices, is_disabled = self._get_topk_indices(scores)
        spkcache, spkcache_preds = self._gather_spkcache_and_preds(emb_seq, preds, topk_indices, is_disabled)
        return spkcache, spkcache_preds, spk_perm
