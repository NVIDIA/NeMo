# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

__all__ = ['SortformerModules']


class SortformerModules(NeuralModule, Exportable):
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
        tf_d_model (int):
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
    def init_weights(self, m):
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
    ):
        super().__init__()
        self.fc_d_model = fc_d_model
        self.tf_d_model = tf_d_model
        self.hidden_size = tf_d_model
        self.unit_n_spks: int = num_spks
        self.hidden_to_spks = nn.Linear(2 * self.hidden_size, self.unit_n_spks)
        self.first_hidden_to_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.single_hidden_to_spks = nn.Linear(self.hidden_size, self.unit_n_spks)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_proj = nn.Linear(self.fc_d_model, self.tf_d_model)

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
        max_len=context_embs.shape[1]
        # create a tensor with the shape (batch_size, 1) filled with ones
        row_vector = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
        # create a tensor with the shape (batch_size, max_len) filled with lengths
        length_matrix = lengths.unsqueeze(1).expand(-1, max_len).to(lengths.device)
        # create a mask by comparing the row vector and length matrix
        mask = row_vector < length_matrix
        return mask.float().to(context_embs.device)
     
    def forward_speaker_sigmoids(self, hidden_out):
        hidden_out = self.dropout(F.relu(hidden_out))
        hidden_out = self.first_hidden_to_hidden(hidden_out)
        hidden_out = self.dropout(F.relu(hidden_out))
        spk_preds = self.single_hidden_to_spks(hidden_out)
        preds = nn.Sigmoid()(spk_preds)
        return preds
