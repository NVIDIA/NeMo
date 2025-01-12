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

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.utils import logging
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import EncodedRepresentation, LengthsType, NeuralType, SpectrogramType
from nemo.core.neural_types.elements import ProbsType

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
        hidden_size: int = 192,
        dropout_rate: float = 0.5,
        fc_d_model: int = 512,
        tf_d_model: int = 192,
        subsampling_factor: int = 8,
        mem_len: int = 188,
        fifo_len: int = 0,  
        step_len: int = 376,
        mem_refresh_rate: int = 1,
        use_memory_pe: bool = False,
        step_left_context: int = 0,
        step_right_context: int = 0,
        mem_sil_frames_per_spk: int = 5,
    ):
        super().__init__()
        self.mem_sil_frames_per_spk = mem_sil_frames_per_spk
        self.step_left_context = step_left_context
        self.step_right_context = step_right_context
        self.subsampling_factor = subsampling_factor
        self.use_memory_pe = use_memory_pe
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

    def length_to_mask(self, context_embs):
        """
        Convert length values to encoder mask input tensor.

        Args:
            lengths (torch.Tensor): tensor containing lengths of sequences
            max_len (int): maximum sequence length

        Returns:
            mask (torch.Tensor): tensor of shape (batch_size, max_len) containing 0's
                                in the padded region and 1's elsewhere
        """
        lengths = torch.tensor([context_embs.shape[1]] * context_embs.shape[0])
        batch_size = context_embs.shape[0]
        max_len = context_embs.shape[1]
        # create a tensor with the shape (batch_size, 1) filled with ones
        row_vector = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
        # create a tensor with the shape (batch_size, max_len) filled with lengths
        length_matrix = lengths.unsqueeze(1).expand(-1, max_len).to(lengths.device)
        # create a mask by comparing the row vector and length matrix
        mask = row_vector < length_matrix
        return mask.float().to(context_embs.device)
    
    def streaming_feat_loader(self, feat_seq):
        """
        Load a chunk of feature sequence for streaming inference.

        Args:
            feat_seq (torch.Tensor): Tensor containing feature sequence
                Dimension: (batch_size, feat_dim, feat frame count)

        Yields:
            step_idx (int): Index of the current step
            chunk_feat_seq (torch.Tensor): Tensor containing the chunk of feature sequence
                Dimension: (batch_size, diar frame count, feat_dim)
            feat_lengths (torch.Tensor): Tensor containing lengths of the chunk of feature sequence
                Dimension: (batch_size,)
        """
        feat_len = feat_seq.shape[2]
        num_chunks = math.ceil(feat_len / (self.step_len * self.subsampling_factor))
        if self.log:
            logging.info(f"feat_len={feat_len}, num_chunks={num_chunks}")
        stt_feat = 0
        for step_idx in range(num_chunks):
            left_offset = min(self.step_left_context * self.subsampling_factor, stt_feat)
            end_feat = min(stt_feat + self.step_len * self.subsampling_factor, feat_len)
            right_offset = min(self.step_right_context * self.subsampling_factor, feat_len-end_feat)
            chunk_feat_seq = feat_seq[:, :, stt_feat-left_offset:end_feat+right_offset]
            stt_feat = end_feat

            feat_lengths = torch.tensor(chunk_feat_seq.shape[-1]).repeat(chunk_feat_seq.shape[0])
            chunk_feat_seq_t = torch.transpose(chunk_feat_seq, 1, 2)
            if self.log:
                logging.info(f"step_idx: {step_idx}, chunk_feat_seq_t shape: {chunk_feat_seq_t.shape}")
            yield step_idx, chunk_feat_seq_t, feat_lengths, left_offset, right_offset
     
    def forward_speaker_sigmoids(self, hidden_out):
        hidden_out = self.dropout(F.relu(hidden_out))
        hidden_out = self.first_hidden_to_hidden(hidden_out)
        hidden_out = self.dropout(F.relu(hidden_out))
        spk_preds = self.single_hidden_to_spks(hidden_out)
        preds = nn.Sigmoid()(spk_preds)
        return preds
    
    def concat_embs(self, list_of_tensors=[], return_lengths=False, dim=1, device=None):
        
        embs = torch.cat(list_of_tensors, dim=dim).to(device) # B x T x D
        lengths = torch.tensor(embs.shape[1]).repeat(embs.shape[0]).to(device)
        
        if return_lengths:
            return embs, lengths    
        else:
            return embs
    
    def init_memory(self, batch_size, d_model=192, device=None):
        return torch.zeros(batch_size, 0, d_model).to(device)

    def update_memory_FIFO(self, mem, fifo, chunk, preds, chunk_left_offset=0, chunk_right_offset=0):
        """
        update the FIFO queue and memory buffer with the chunk of embeddings and speaker predictions
        Args:
            mem (torch.Tensor): memory buffer to save the embeddings from start
                Dimension: (batch_size, mem_len, emb_dim)
            fifo (torch.Tensor): FIFO queue to save the embeddings from the latest chunks 
                Dimension: (batch_size, fifo_len, emb_dim)
            chunk (torch.Tensor): chunk of embeddings to be predicted
                Dimension: (batch_size, chunk_len, emb_dim)
            preds (torch.Tensor): speaker predictions of the [mem + fifo + chunk] embeddings
                Dimension: (batch_size, mem_len + fifo_len + chunk_len, num_spks)
            chunk_left_offset and chunk_right_offset (int): left & right offset of the chunk,
                only the chunk[:, chunk_left_offset:chunk_len+chunk_left_offset] is used for FIFO queue

        Returns:
            mem (torch.Tensor): updated memory buffer
                Dimension: (batch_size, mem_len, emb_dim)
            fifo (torch.Tensor): updated FIFO queue
                Dimension: (batch_size, fifo_len, emb_dim)
            chunk_preds (torch.Tensor): speaker predictions of the chunk embeddings
                Dimension: (batch_size, chunk_len, num_spks)
        """

        B, T, D = mem.shape

        lc, rc = chunk_left_offset, chunk_right_offset
        mem_len, fifo_len, chunk_len = mem.shape[1], fifo.shape[1], chunk.shape[1] - lc - rc

        mem_preds, fifo_preds = preds[:, :mem_len], preds[:, mem_len:mem_len + fifo_len]
        chunk = chunk[:, lc:chunk_len + lc]
        chunk_preds = preds[:, mem_len + fifo_len + lc:mem_len + fifo_len + chunk_len + lc]

        if self.fifo_len == 0:
            assert fifo_len == self.fifo_len
            pop_out_embs, pop_out_preds = chunk, chunk_preds
        else:
            fifo = torch.cat([fifo, chunk], dim=1)
            if fifo.size(1) <= self.fifo_len:
                pop_out_embs, pop_out_preds = self.init_memory(B, D, mem.device), self.init_memory(B, self.unit_n_spks, mem.device)
            else:
                if self.mem_refresh_rate == 0: # clear fifo queue when it reaches the max_fifo_len and update memory buffer
                    pop_out_embs  = fifo[:, :fifo_len]
                    pop_out_preds = fifo_preds
                    fifo = self.init_memory(B, D, mem.device)
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
            if mem.shape[1] > self.mem_len:
                mem = self._compress_memory(mem, torch.cat([mem_preds, pop_out_preds], dim=1))
            
        if self.log:
            logging.info(f"MC mem: {mem.shape}, chunk: {chunk.shape}, fifo: {fifo.shape}, chunk_preds: {chunk_preds.shape}")
            
        return mem, fifo, mem_preds, fifo_preds, chunk_preds
    
    def _compress_memory(self, emb_seq, preds):
        """.
        Compresses memory for streaming inference
        Keeps mem_len most important frames out of input n_frames, based on speaker sigmoid scores and positional information

        Args:
            emb_seq (torch.Tensor): Tensor containing n_frames > mem_len (mem+chunk) embeddings
                Dimension: (batch_size, n_frames, emb_dim)
            preds (torch.Tensor): Tensor containing n_frames > mem_len (mem+chunk) speaker sigmoid outputs
                Dimension: (batch_size, n_frames, max_num_spk)

        Returns:
            memory_buff (torch.Tensor): concatenation of num_spk subtensors of emb_seq corresponding to each speaker
            each of subtensors contains (mem_len//num_spk) frames out of n_frames
                Dimension: (batch_size, mem_len, emb_dim)
        """
        B, n_frames, n_spk = preds.shape
        emb_dim = emb_seq.shape[2]
        mem_len_per_spk = self.mem_len // n_spk
        last_n_sil_per_spk = self.mem_sil_frames_per_spk

        #condition for frame being silence
        is_sil = preds.sum(dim=2) < 0.1 # Shape: (B, n_frames)
        is_sil = is_sil.unsqueeze(-1) # Shape: (B, n_frames, 1)
        #get mean silence embedding tensor
        emb_seq_sil = torch.where(is_sil, emb_seq, torch.tensor(0.0)) # Shape: (B, n_frames, emb_dim)
        emb_seq_sil_sum = emb_seq_sil.sum(dim=1) # Shape: (B, emb_dim)
        sil_count = is_sil.sum(dim=1).clamp(min=1) # Shape: (B)
        emb_seq_sil_mean = emb_seq_sil_sum / sil_count # Shape: (B, emb_dim)
        emb_seq_sil_mean = emb_seq_sil_mean.unsqueeze(1).expand(-1, n_spk*mem_len_per_spk, -1) # Shape: (B, n_spk*mem_len_for_spk, emb_dim)

        if self.use_memory_pe:
            #add position embeddings
            start_pos=0
            position_ids = torch.arange(start=start_pos, end=start_pos + n_frames, dtype=torch.long, device=preds.device)
            position_ids = position_ids.unsqueeze(0).repeat(preds.size(0), 1)
            preds = preds + self.memory_position_embedding(position_ids)

        #get frame importance scores
        encoder_mask = self.length_to_mask(preds)
        # scores = self.transformer_memory_compressor(encoder_states=preds, encoder_mask=encoder_mask) # Shape: (B, n_frames, n_spk)
        scores = preds
#        logging.info(f"MC scores: {scores[0,:,:]}")

        #normalized scores (non-overlapped frames are more preferable for memory)
        scores_norm = 2*scores - torch.sum(scores, dim=2).unsqueeze(-1).expand(-1, -1, n_spk)
#        logging.info(f"MC scores normalized: {scores_norm[0,:,:]}")

        #cumsum-normalized scores: this is to avoid speakers appearing in memory buffer before their block
        # as a result, for speaker i={0,1,2,...,n_spk-1}, scores_csnorm_i = 2*scores_i - sum_{j=i}^{n_spk-1}(scores_j)
        scores_csnorm = 2*scores - scores.cpu().flip(dims=[2]).cumsum(dim=2).flip(dims=[2]).to(preds.device) #ugly hack to ensure deterministic behavior
#        logging.info(f"MC scores cumsum-normalized: {scores_csnorm[0,:,:]}")

        #scores thresholding: set -inf if cumsum-normalized score is less than 0.5.
        # This exclude non-speech frames and also doesn't allow speakers to appear in memory before their block
        is_good = scores_csnorm > 0.5
        scores = torch.where(is_good, scores_norm, torch.tensor(float('-inf'))) # Shape: (B, n_frames, n_spk)

        #ALTERNATIVE: thresholding, then using frame index as a score to keep latest possible frames in memory
#        scores = torch.where(is_good, torch.arange(n_frames, device=scores.device).view(1, n_frames, 1), torch.tensor(float('-inf'))) # Shape: (B, n_frames, n_spk)

#        logging.info(f"MC scores final: {scores[0,:,:]}")

        #get mem_len_per_spk most important indices for each speaker
        topk_values, topk_indices = torch.topk(scores, mem_len_per_spk-last_n_sil_per_spk, dim=1, largest=True, sorted=False) # Shape: (B, mem_len_per_spk-last_n_sil_per_spk, n_spk)
        valid_topk_mask = topk_values != float('-inf')
        topk_indices = torch.where(valid_topk_mask, topk_indices, torch.tensor(9999))  # Replace invalid indices with 9999

        if last_n_sil_per_spk > 0: #add number of silence frames in the end of each block
            topk_indices = torch.cat([topk_indices, torch.full((B, last_n_sil_per_spk, n_spk), 9999, device=topk_indices.device)], dim=1) # Shape: (B, mem_len_for_spk, n_spk)

        topk_indices = topk_indices.permute(0, 2, 1) # Shape: (B, n_spk, mem_len_for_spk)

        # sort indices to preserve original order of frames
        topk_indices_sorted, _ = torch.sort(topk_indices, dim=2) # Shape: (B, n_spk, mem_len_for_spk)
        topk_indices_flatten = topk_indices_sorted.reshape(B, n_spk*mem_len_per_spk) # Shape: (B, n_spk*mem_len_for_spk)
#        logging.info(f"MC topk indices: {topk_indices_sorted[0,:,:]}")

        # condition of being invalid index
        is_inf = topk_indices_flatten == 9999
        topk_indices_flatten[is_inf] = 0 # set a placeholder index instead of 9999 to make gather work

        # expand topk indices to emb_dim in last dimension to use gather
        topk_indices_expanded = topk_indices_flatten.unsqueeze(-1).expand(-1, -1, emb_dim) # Shape: (B, n_spk*mem_len_for_spk, emb_dim)

        # gather memory buffer including placeholder embeddings for silence frames
        emb_seq_gathered = torch.gather(emb_seq, 1, topk_indices_expanded) # Shape: (B, n_spk*mem_len_for_spk, emb_dim)

        # replace placeholder embeddings with actual mean silence embedding
        memory_buff = torch.where(is_inf.unsqueeze(-1), emb_seq_sil_mean, emb_seq_gathered)

        return memory_buff
    
    def visualize_all_session(self, mem_preds_list, fifo_preds_list, chunk_preds_list, uniq_ids=None, out_dir='./'):
        '''
        Visualize the memory, FIFO queue, and chunk predictions for all sessions.
        Each session is saved as a gif file.
        Args:
            mem_preds_list: list of torch.Tensor, [(B, mem_len, num_spks)]
            fifo_preds_list: list of torch.Tensor, [(B, fifo_len, num_spks)]
            chunk_preds_list: list of torch.Tensor, [(B, chunk_len, num_spks)]
        '''
        batch_size = mem_preds_list[0].shape[0]
        if uniq_ids is None:
            uniq_ids = [f'batch_{i:05d}' for i in range(batch_size)]

        pbar = tqdm(enumerate(uniq_ids), total=len(uniq_ids))
        for i, uniq_id in pbar:
            pbar.set_description(f'Visualizing session {uniq_id}')
            mem_preds = [mem_preds_list[j][i] for j in range(len(mem_preds_list))]
            fifo_preds = [fifo_preds_list[j][i] for j in range(len(fifo_preds_list))]
            chunk_preds = [chunk_preds_list[j][i] for j in range(len(chunk_preds_list))]

            pbar.set_description(f'Generating frames for session {uniq_id}')
            frames = self.preds_to_frames(mem_preds, fifo_preds, chunk_preds)
            
            pbar.set_description(f'Saving GIF for session {uniq_id}')
            self.array_to_gif(frames, f'{out_dir}/{uniq_id}.gif', frame_len=self.step_len*80)

            
    
    def preds_to_frames(self, mem_preds, fifo_preds, chunk_preds):
        '''
        Generate frames from the memory, FIFO queue, and chunk predictions from one session.
        Args:
            mem_preds: list of torch.Tensor, [(mem_len, num_spks)]
            fifo_preds: list of torch.Tensor, [(fifo_len, num_spks)]
            chunk_preds: list of torch.Tensor, [(chunk_len, num_spks)]
        '''
        n_steps = len(mem_preds)
        frames = []
        for step_idx in range(n_steps):
            frame = self.pred_to_frame(mem_preds[step_idx], fifo_preds[step_idx], chunk_preds[step_idx], step_idx)
            frames.append(frame)

        return frames

    
    def pred_to_frame(self, mem_pred, fifo_pred, chunk_pred, step_idx=0):
        '''
        Generate a frame from the memory, FIFO queue, and chunk predictions at one step.
        Args:
            mem_pred: torch.Tensor, (mem_len, num_spks)
            fifo_pred: torch.Tensor, (fifo_len, num_spks)
            chunk_pred: torch.Tensor, (chunk_len, num_spks)
            step_idx: int, the index of the step
        '''
        if mem_pred.shape[0] < self.mem_len:
            mem_pred = np.pad(mem_pred, ((self.mem_len - mem_pred.shape[0], 0), (0, 0)), 'constant', constant_values=0)
        if fifo_pred.shape[0] < self.fifo_len:
            fifo_pred = np.pad(fifo_pred, ((self.fifo_len - fifo_pred.shape[0], 0), (0, 0)), 'constant', constant_values=0)

        cmap_str = 'viridis'
        aspect_float = 7.0
        FS = 20
        offset = round(step_idx * self.step_len * 0.08, 2) 

        # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(40,3), layout='constrained', width_ratios=[10, 10, 1], height_ratios=[1])

        # Plot preds_mat
        cax0 = axs[0].imshow(mem_pred.T, cmap=cmap_str, interpolation='nearest', aspect=aspect_float)
        cax2 = axs[1].imshow(fifo_pred.T, cmap=cmap_str, interpolation='nearest', aspect=aspect_float)
        cax3 = axs[2].imshow(chunk_pred.T, cmap=cmap_str, interpolation='nearest', aspect=aspect_float)
        axs[0].set_title('Memory', fontsize=FS)
        axs[1].set_title('FIFO queue', fontsize=FS)
        minutes = int(offset // 60)
        seconds = offset % 60
        axs[2].set_title(f'Chunk {minutes:02}:{seconds:05.2f}', fontsize=FS)

        for a in axs:
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame = Image.fromarray(frame)
        plt.close()

        return frame

    def array_to_gif(self, frames, filepath, frame_len=960):
        '''
        Save a list of frames as a gif file.
        Args:
            frames: list of PIL.Image, [frame1, frame2, ...]
            filepath: str, the path to save the gif file
            frame_len: int, the duration in milliseconds per frame
        '''
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:],
            duration=frame_len,  # Duration in milliseconds per frame
            loop=0  # Loop forever
        )
        
