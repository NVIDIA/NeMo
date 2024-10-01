#            preds_mean = preds Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import EncodedRepresentation, LengthsType, NeuralType, SpectrogramType
from nemo.core.neural_types.elements import ProbsType

__all__ = ['SortformerDiarizer']


class SortformerDiarizer(NeuralModule, Exportable):
    """
    Multi-scale Diarization Decoder (MSDD) for overlap-aware diarization and improved diarization accuracy from clustering diarizer.
    Based on the paper: Taejin Park et. al, "Multi-scale Speaker Diarization with Dynamic Scale Weighting", Interspeech 2022.
    Arxiv version: https://arxiv.org/pdf/2203.15974.pdf

    Args:
        num_spks (int):
            Max number of speakers that are processed by the model. In `MSDD_module`, `num_spks=2` for pairwise inference.
        hidden_size (int):
            Number of hidden units in sequence models and intermediate layers.
        num_lstm_layers (int):
            Number of the stacked LSTM layers.
        dropout_rate (float):
            Dropout rate for linear layers, CNN and LSTM.
        # cnn_output_ch (int):
        #     Number of channels per each CNN layer.
        emb_dim (int):
            Dimension of the embedding vectors.
        scale_n (int):
            Number of scales in multi-scale system.
        clamp_max (float):
            Maximum value for limiting the scale weight values.
        conv_repeat (int):
            Number of CNN layers after the first CNN layer.
        weighting_scheme (str):
            Name of the methods for estimating the scale weights.
        context_vector_type (str):
            If 'cos_sim', cosine similarity values are used for the input of the sequence models.
            If 'elem_prod', element-wise product values are used for the input of the sequence models.
    """

    @property
    def output_types(self):
        """
        Return definitions of module output ports.
        """
        return OrderedDict(
            {
                "probs": NeuralType(('B', 'T', 'C'), ProbsType()),
                "scale_weights": NeuralType(('B', 'T', 'C', 'D'), ProbsType()),
            }
        )

    @property
    def input_types(self):
        """
        Return  definitions of module input ports.
        """
        return OrderedDict(
            {
                "ms_emb_seq": NeuralType(('B', 'T', 'C', 'D'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
                "ms_avg_embs": NeuralType(('B', 'C', 'D', 'C'), EncodedRepresentation()),
            }
        )

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.01)

    def __init__(
        self,
        num_spks: int = 4,
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        dropout_rate: float = 0.5,
        cnn_output_ch: int = 16,
        emb_dim: int = 192,
        scale_n: int = 5,
        clamp_max: float = 1.0,
        conv_repeat: int = 1,
        use_amsl_layer: bool = False,
        weighting_scheme: str = 'conv_scale_weight',
        context_vector_type: str = 'cos_sim',
        sort_layer_on: bool = False, 
        sort_final_layer_on: bool = False,
        merge_unit_count: int = 1,
        mem_len: int = 2400,
        step_len: int = 2400,
    ):
        super().__init__()
        self._speaker_model = None
        self.sort_final_layer_on = sort_final_layer_on
        # self.length: int = 50
        self.emb_dim: int = emb_dim
        self.num_spks: int = num_spks
        self.unit_n_spks: int = num_spks
        if self.num_spks % self.unit_n_spks != 0:
            raise ValueError("The number of speakers must be divisible by unit_n_spks.")
        elif self.num_spks > self.unit_n_spks:
            self.n_stream = self.num_spks // self.unit_n_spks
        else:
            self.n_stream = 1
        
        self.scale_n: int = scale_n
        self.num_lstm_layers: int = num_lstm_layers
        self.hidden_to_spks = nn.Linear(2 * hidden_size, self.unit_n_spks)
        self.first_hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.target_to_embs = nn.Linear(hidden_size, hidden_size)
       
        self.W_a = nn.Linear(emb_dim, emb_dim, bias=False)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate,
        ) 
        self.single_hidden_to_spks = nn.Linear(hidden_size, self.unit_n_spks)
        self.dist_to_emb = nn.Linear(self.scale_n * self.unit_n_spks, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_to_spks.apply(self.init_weights)
        self.lstm.apply(self.init_weights)
        
        # Streaming Sortformer parameters 
        self.mem_len = mem_len
        self.step_len = step_len
        self.embedding_list = []
        self.eps = 1e-6
        self.memory_compress_pool_size = 2
     
    def _zero_sub_diag(self, trans_mat, emb_win:int=10, full_upper=False):
        """
        Zeroes out elements below a certain diagonal in a matrix.

        This method creates a mask to zero out elements below a specified diagonal in the input matrix `trans_mat`.
        If `emb_win` is greater than 0, it will shift `emb_win` number of steps above the diagonal line.

        Args:
            trans_mat (torch.Tensor): 
                The input matrix to be masked. 
                Dimension: (batch_size, step_len, step_len)
            emb_win (int): 
                The number of steps above the diagonal line to start zeroing out elements. Default is 10.
            full_upper (bool): 
                If True, the entire upper triangular part of the matrix is retained. Default is False.

        Returns:
            total_mask(torch.Tensor): 
                The masked matrix with elements below the specified diagonal zeroed out.
        """
        mask1 = torch.triu(torch.ones_like(trans_mat[0], dtype=torch.bool),  diagonal=0).to(trans_mat.device)
        mask2 = torch.triu(torch.ones_like(trans_mat[0], dtype=torch.bool),  diagonal=-1*emb_win).T.to(trans_mat.device)
        if full_upper:
            ext_mask = mask1
        else:
            ext_mask = (mask1 * mask2).unsqueeze(0).repeat(trans_mat.shape[0], 1, 1)
        total_mask = trans_mat * ext_mask
        return total_mask
    
    def _attention_score_compressor(
        self, 
        batch_size: int, 
        mem_and_new_embs: torch.Tensor, 
        compress_ratio: int,
        ):
        """
        Use attention score to compress the new incoming embeddings with the memory embeddings.

        Args:
            batch_size (int):
                Batch size of the infereced embeddings.
            mem_and_new_embs (torch.Tensor):
                Concatenated memory embeddings and new incoming embeddings.
                Dimension: [batch_size, mem_len + step_len, emb_dim]
            compress_ratio (int):
                The ratio of compressing the new incoming embeddings.
                
        Returns:
            assigner_mat (torch.Tensor):
                Assigner matrix that selects the embedding vectors need to be removed.
                Dimension: [batch_size, mem_len, (mem_len+step_len)]
        """
        # Create a block matrix of all-ones matrices.
        selection_mask = torch.eye(self.step_len).repeat(batch_size, compress_ratio, compress_ratio)  # [batch_size, (mem_len+step_len), (mem_len+step_len)]
        total_len_eye = torch.eye(self.step_len+self.mem_len).unsqueeze(0).repeat(batch_size, 1, 1).to(mem_and_new_embs.device)  # [batch_size, (mem_len+step_len), (mem_len+step_len)]
        # Calculate the attention score between the memory embeddings and the new incoming embeddings.
        batch_attention_raw = torch.bmm(mem_and_new_embs, mem_and_new_embs.transpose(1, 2)).cpu()  # [batch_size, (mem_len+step_len), (mem_len+step_len)]
        batch_attention = selection_mask * batch_attention_raw  # [batch_size, (mem_len+step_len), (mem_len+step_len)]
        # For every `compress_ratio` rows, find a row with the highest attention score, which means the most redundant sample.
        batch_unit_win_max_inds = batch_attention.sum(dim=2).reshape(batch_size, -1, compress_ratio).max(dim=2)[1]  # [batch_size, step_len]
        batch_target_remove_inds = batch_unit_win_max_inds + (torch.arange(self.step_len) * compress_ratio).unsqueeze(0).repeat(batch_size, 1) # [batch_size, step_len]
        batch_inds_for_masker = torch.arange(batch_size).unsqueeze(1).expand_as(batch_target_remove_inds)  # [batch_size, step_len]
        # Create a masker matrix that selects the embedding vectors need to be removed.
        masker = torch.ones((batch_size, self.mem_len+self.step_len)).bool().to(mem_and_new_embs.device)  # [batch_size, (mem_len+step_len)]
        masker[batch_inds_for_masker, batch_target_remove_inds] = False  # [batch_size, (mem_len+step_len)]
        assinger_mat = total_len_eye.view(batch_size * (self.mem_len+self.step_len), -1)[masker.view(-1), :].reshape(batch_size, self.mem_len, -1) # [batch_size, mem_len, mem_len + step_len]
        return assinger_mat
    
    def _drop_to_compress(
        self, 
        chunk_emb_seq: torch.Tensor, 
        mem_and_new_embs: torch.Tensor, 
        mem_and_new_labels: torch.Tensor, 
        compress_ratio: int,
        thres: float =0.95,
        ):
        """
        Compress the memory and new embeddings by dropping less significant embeddings based on a threshold.

        Args:
            chunk_emb_seq (torch.Tensor): 
                The chunk of embedding sequences. Dimension: [batch_size, step_len, emb_dim].
            mem_and_new_embs (torch.Tensor): 
                Concatenated memory embeddings and new incoming embeddings. 
                Dimension: [batch_size, mem_len + step_len, emb_dim].
            mem_and_new_labels (torch.Tensor): 
                Concatenated memory labels and new incoming labels. 
                Dimension: [batch_size, mem_len + step_len, label_dim].
            compress_ratio (int): 
                The ratio of compressing the new incoming embeddings.
            thres (float): 
                Threshold for determining significant embeddings. Default is 0.95.

        Returns:
            assinger_mat (torch.Tensor): 
                Assigner matrix that selects the embedding vectors to be removed. 
                Dimension: [batch_size, compressed_len, (mem_len+step_len)].
        """
        # Remove the labels that are classified as overlaps.
        overlap_bool_nl = mem_and_new_labels.sum(dim=2) > 1  # [batch_size, (mem_len+step_len)]
        mnl_noovl = mem_and_new_labels.clone()  # [batch_size, (mem_len+step_len), label_dim]
        mnl_noovl[overlap_bool_nl] = 0  # [batch_size, (mem_len+step_len), label_dim]
        # Include silence to the label and add another dimension for silence
        sil_bools = (mnl_noovl.sum(dim=2) == 0)  # [batch_size, (mem_len+step_len)]
        silaug_mnl_noovl = torch.cat((mnl_noovl, torch.zeros_like(mnl_noovl[:, :, 0].unsqueeze(2))), dim=2)  # [batch_size, (mem_len+step_len), label_dim + 1]
        silaug_mnl_noovl[torch.arange(sil_bools.shape[0]).unsqueeze(1), :, -1].squeeze(1)[sil_bools] = 1  # [batch_size, (mem_len+step_len), label_dim + 1]
        trans_label_mask = torch.bmm(silaug_mnl_noovl, silaug_mnl_noovl.transpose(1, 2))  # [batch_size, (mem_len+step_len), (mem_len+step_len)]
        trans_label_mask_zeroed = self._zero_sub_diag(trans_label_mask, emb_win=0).detach().cpu()  # [batch_size, (mem_len+step_len), (mem_len+step_len)]
        # Create compression mask matrix that selects ones and zeros
        compression_mask_inds = (torch.arange(trans_label_mask.size(1)) % compress_ratio != 1).cpu()  # [(mem_len+step_len)]
        trans_label_mask_zerocomp = trans_label_mask_zeroed[:, compression_mask_inds, :]  # [batch_size, compressed_len, (mem_len+step_len)]
        row_maxs = trans_label_mask_zerocomp.max(dim=2, keepdim=True)[0]  # [batch_size, compressed_len, 1]
        normalized_compress_mat = trans_label_mask_zerocomp / (row_maxs + self.eps)  # [batch_size, compressed_len, (mem_len+step_len)]
        assinger_mat = torch.zeros_like(normalized_compress_mat).to(chunk_emb_seq.device)  # [batch_size, compressed_len, (mem_len+step_len)]
        assinger_mat[(normalized_compress_mat >= thres)] = 1  # [batch_size, compressed_len, (mem_len+step_len)]
        row_maxs_post = assinger_mat.sum(dim=2).unsqueeze(2)  # [batch_size, compressed_len, 1]
        assinger_mat = assinger_mat / (row_maxs_post + self.eps)  # [batch_size, compressed_len, (mem_len+step_len)]
        return assinger_mat
    
    def memory_compressor(
        self,
        step_idx: int,
        chunk_emb_seq: torch.Tensor,
        prev_preds: torch.Tensor,
        memory_buff: torch.Tensor,
        memory_label: torch.Tensor,
        compress_rate: int = 2,
        UNIT_LEN: int = 10,
        use_attn_score: bool = True,
        ):
        """

        Args:
            chunk_emb_seq:
                Dimension: (batch_size, step_len, emb_dim)
            prev_preds:
                Dimension: (batch_size, max_spks, step_len)
            memory_buff:
                Dimension: (batch_size, mem_len, emb_dim)
            memory_label:
                Dimension: (batch_size, max_spks, mem_len)
        """
        chunk_emb_seq = chunk_emb_seq.detach() if chunk_emb_seq is not None else None
        memory_buff = memory_buff.detach() if memory_buff is not None else None
        memory_label = memory_label.detach() if memory_label is not None else None
        mem_int, mnl_int = int(self.mem_len/UNIT_LEN), int(self.step_len/UNIT_LEN)
        compress_ratio = (mem_int+mnl_int)//mnl_int
        if step_idx == 0:
            # First trial, we only calculate mem_len 
            new_memory_buff = memory_buff # [bs, step_len, emb_dim]
            new_memory_label = prev_preds # [bs, step_len]
        else:
            batch_size = chunk_emb_seq.shape[0]
            new_preds = prev_preds[:, self.mem_len:, :] # [batch_size, max_spks, step_len]
            # Prepare repeated mem_and_new_labels
            mem_and_new_labels = torch.cat([memory_label, new_preds], dim=1) # [batch_size, (mem_len + step_len), max_spks]
            mem_and_new_embs = torch.cat([memory_buff, chunk_emb_seq], dim=1) # [batch_size, (mem_len + step_len), step_len]
            if use_attn_score:
                assinger_mat = self._attention_score_compressor(batch_size, mem_and_new_embs, compress_ratio)
            else:
                assinger_mat = self._drop_to_compress(chunk_emb_seq, mem_and_new_embs, mem_and_new_labels, compress_ratio)
            assinger_mat = assinger_mat.to(mem_and_new_embs.device) 
            new_memory_buff = torch.bmm(assinger_mat, mem_and_new_embs)
            new_memory_label = torch.bmm(assinger_mat, mem_and_new_labels).bool().float() # [batch_size, max_spks, (mem_len + step_len)]
        return new_memory_buff, new_memory_label
 

    def forward_context(self, ms_emb_seq, length, ms_avg_embs):
        context_emb, scale_weights = self.core_model(ms_emb_seq, length, ms_avg_embs)
        context_emb = self.dropout(F.relu(context_emb))
        return context_emb, scale_weights
    
    def sort_output_states(self, output_states):
        bs, seq_len, emb_dim = output_states.shape
        sorted_output_states_list = []
        per_sample_vars = torch.var(output_states, dim=1)
        sort_indices = torch.sort(per_sample_vars, descending=True)[1]
        for i in range(bs):
            sorted_output_states_list.append(output_states[i, :, sort_indices[i]].unsqueeze(0))
        sorted_output_states = torch.concat(sorted_output_states_list)
        return sorted_output_states
     
    def forward_speaker_sigmoids(self, hidden_out):
        if self.sort_final_layer_on:
            hidden_out = self.sort_output_states(hidden_out)
        hidden_out = self.dropout(F.relu(hidden_out))
        hidden_out = self.first_hidden_to_hidden(hidden_out)
        if self.sort_final_layer_on:
            hidden_out = self.sort_output_states(hidden_out)
        hidden_out = self.dropout(F.relu(hidden_out))
        spk_preds = self.single_hidden_to_spks(hidden_out)
        preds = nn.Sigmoid()(spk_preds)
        return preds
    
    def mock_embedding(self, temp_targets, emb_dim=192):
        emb_dim, offset_val = 192, 1e+3
        emb_seq = torch.repeat_interleave(temp_targets, int(emb_dim/temp_targets.shape[-1]) , dim=2)
        if self.cfg_e2e_diarizer_model.variance_in_mock_embs > 0:
            random_embs = torch.randn(emb_seq.shape[0], emb_seq.shape[-1]).unsqueeze(1).repeat(1, emb_seq.shape[1], 1).to(emb_seq.device)
            emb_seq = (offset_val * emb_seq - self.cfg_e2e_diarizer_model.variance_in_mock_embs * random_embs)/offset_val
        emb_seq = emb_seq + torch.randn_like(emb_seq) * self.cfg_e2e_diarizer_model.mock_emb_noise_std 
        mock_embs = self.target_to_embs(emb_seq) 
        return mock_embs 
    
    def input_example(self, lens=100):
        """
        Generate input examples for tracing etc.

        Returns (tuple):
            A tuple of input examples.
        """
        device = next(self.parameters()).device
        input_example = torch.randn(1, lens, self.scale_n, self.emb_dim, device=device)
        lens = torch.full(size=(input_example.shape[0],), fill_value=123, device=device)
        avg_embs = torch.randn(1, self.scale_n, self.emb_dim, self.num_spks, device=device)
        return tuple([input_example, lens, avg_embs])
