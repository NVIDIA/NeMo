# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LogprobsType, NeuralType
from typing import Dict, List, Optional, Union
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    LengthsType,
    LogitsType,
    LogprobsType,
    EncodedRepresentation,
    NeuralType,
    SpectrogramType,
)
from nemo.core.neural_types.elements import ProbsType

def sprint(*args):
    if False:
    # if True:
        print(*args)
    else:
        pass

__all__ = ['SampleConvASREncoder', 'LSTMDecoder', 'MSDD_module']

class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=(3, 1), stride=(1, 1)):
        super(ConvLayer, self).__init__()
        pad_size = (kernel_size[1]-1)//2
        self.cnn = nn.Sequential(
                      nn.Conv2d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=kernel_size, 
                                stride=stride),
                      nn.ReLU(),
                      nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
                   )

    def forward(self, feature):
        feature = self.cnn(feature)
        return feature
     
class MSDD_module(NeuralModule, Exportable):
    """
    Multiscale Diarization Decoder for overlap-aware diarization.
    Args:
        feat_in (int): size of the input features
        num_speakers(int): the number of speakers
        scale_weights:  batch_size x lengths x scale_n
    """

    @property
    def output_types(self):
        return OrderedDict(
                {
                "probs": NeuralType(('B', 'T', 'C'), ProbsType()),
                "scale_weights": NeuralType(('B', 'T', 'C'), ProbsType())
                }
            )
    
    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return OrderedDict(
            {
            "ms_emb_seq": NeuralType(('B', 'T', 'C', 'D'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
            "ms_avg_embs": NeuralType(('B', 'C', 'D', 'C'), EncodedRepresentation()),
            "targets": NeuralType(('B', 'T', 'C'), ProbsType()),
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
            num_spks: int = -1,
            hidden_size=256,
            num_lstm_layers=2,
            dropout_rate: float=0.5,
            cnn_output_ch=4,
            emb_sizes=192,
            scale_n: int=1):
        super().__init__()
        
        # self._speaker_model = None
        self.emb_sizes = emb_sizes
        self.num_spks = num_spks
        self.scale_n = scale_n
        self.cnn_output_ch = cnn_output_ch
        self.chan = 1
        self.eps = 1e-6
        self.num_lstm_layers = num_lstm_layers
        self.fixed_ms_weight = torch.ones(self.scale_n)/self.scale_n
        self.softmax = torch.nn.Softmax(dim=2)
        self.cos_dist = torch.nn.CosineSimilarity(dim=3, eps=self.eps)
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=self.num_lstm_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.conv1 = ConvLayer(in_channels=1, out_channels=cnn_output_ch, kernel_size=(self.scale_n + self.scale_n*num_spks, 1), stride=(1, 1))
        self.conv2 = ConvLayer(in_channels=1, out_channels=cnn_output_ch, kernel_size=(self.cnn_output_ch, 1), stride=(1, 1))
        
        self.conv_to_linear = nn.Linear(emb_sizes*cnn_output_ch, hidden_size)
        self.linear_to_weights = nn.Linear(hidden_size, self.scale_n)
        self.hidden_to_spks = nn.Linear(2*hidden_size, self.num_spks)
        self.dist_to_emb = nn.Linear(self.scale_n * self.num_spks, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.hidden_to_spks.apply(self.init_weights)
        self.dist_to_emb.apply(self.init_weights)
        self.lstm.apply(self.init_weights)
    
    def core_model(self, ms_emb_seq, length, ms_avg_embs, targets):
        """
        Core model that accepts multi-scale cosine similarity values and estimates per-speaker binary label.

        Args:
            targets: ground-truth labels for the finest segment.
                batch_size x feats_len x max_spks
            avg_embs: cluster-average speaker embedding vectors.
                batch_size x scale_n x emb_dim x max_spks
            ms_emb_seq : Multiscale input embedding sequence
                batch_size x feats_len x scale_n x emb_dim

        Returns:
            preds: Predicted binary speaker label for each speaker.
            scale_weights: Multiscale weights per each base-scale segment.

        """
        batch_size = ms_emb_seq.shape[0]
        length = ms_emb_seq.shape[1]
        emb_dim = ms_emb_seq.shape[-1]
        
        _ms_emb_seq = ms_emb_seq.unsqueeze(4).expand(-1, -1, -1, -1, self.num_spks)
        ms_emb_seq_single = ms_emb_seq
        ms_avg_embs = ms_avg_embs.unsqueeze(1).expand(-1, length, -1, -1, -1)

        ms_avg_embs_perm = ms_avg_embs.permute(0, 1, 2, 4, 3).reshape(batch_size, length, -1, emb_dim)
        cos_dist_seq = self.cos_dist(_ms_emb_seq, ms_avg_embs)
        ms_cnn_input_seq = torch.cat([ms_avg_embs_perm, ms_emb_seq_single], dim=2)
        ms_cnn_input_seq = ms_cnn_input_seq.unsqueeze(2).flatten(0, 1)
        
        conv_out1 = self.conv1(ms_cnn_input_seq)
        conv_out1 = self.dropout(conv_out1)
        conv_out1 = conv_out1.reshape(batch_size, length, self.cnn_output_ch, emb_dim)
        conv_out1 = conv_out1.unsqueeze(2).flatten(0, 1)
        conv_out1 = self.dropout(F.relu(conv_out1))

        conv_out2 = self.conv2(conv_out1)
        conv_out2 = conv_out2.permute(0,2,1,3)
        conv_out2 = conv_out2.reshape(batch_size, length, self.cnn_output_ch, emb_dim)
        conv_out2 = self.dropout(F.relu(conv_out2))

        lin_input_seq = conv_out2.view(batch_size, length, self.cnn_output_ch * emb_dim)
        hidden_seq = self.conv_to_linear(lin_input_seq)
        scale_weights = self.linear_to_weights(hidden_seq)
        scale_weights = self.softmax(scale_weights)

        scale_weights = scale_weights.to(ms_emb_seq.device)
        scale_weight_expanded = scale_weights.unsqueeze(3).expand(-1, -1, -1, self.num_spks)
        weighted_cos_dist_seq = torch.mul(scale_weight_expanded, cos_dist_seq)
        seq_input_sum = weighted_cos_dist_seq.sum(axis=2).view(batch_size, length, -1)

        weighted_cos_dist_seq = weighted_cos_dist_seq.view(batch_size, length, -1)
        lstm_input = self.dist_to_emb(weighted_cos_dist_seq)
        lstm_input = self.dropout(F.relu(lstm_input))

        lstm_output, (hn, cn)= self.lstm(lstm_input)
        lstm_hidden_out = self.dropout(F.relu(lstm_output))

        spk_preds = self.hidden_to_spks(lstm_hidden_out)
        preds = nn.Sigmoid()(spk_preds)
        return preds, scale_weights

    @typecheck()
    def forward(self, ms_emb_seq, length, ms_avg_embs, targets):
        preds, scale_weights = self.core_model(ms_emb_seq, length, ms_avg_embs, targets)
        return preds, scale_weights

    def input_example(self):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        device = next(self.parameters()).device
        lens = torch.full(size=(input_example.shape[0],), fill_value=123, device=device)
        input_example = torch.randn(1, lens, self.scale_n, self.emb_sizes, device=device)
        avg_embs = torch.randn(1, self.scale_n, self.emb_sizes, self.num_spks, device=device)
        targets = torch.randn(1, lens, self.num_spks).round().float()
        return tuple([input_example, lens, avg_embs, targets])
