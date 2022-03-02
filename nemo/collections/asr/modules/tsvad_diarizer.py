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

__all__ = ['SampleConvASREncoder', 'LSTMDecoder', 'TSVAD_module']

class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=(3, 1), stride=(0, 1)):
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
     
# class LSTMP(nn.Module):
    # def __init__(self, n_in, n_hidden, nproj, rproj=100, dropout=0, num_layers=1):
        # super(LSTMP, self).__init__()

        # self.num_layers = num_layers

        # self.rnns = nn.ModuleList([nn.LSTM(n_in, n_hidden, bidirectional=False, dropout=dropout, batch_first=True)])
        # self.linears = nn.ModuleList([nn.Linear(n_hidden, nproj)])

        # for i in range(num_layers-1):
            # self.rnns.append(nn.LSTM(nproj, n_hidden, bidirectional=False, dropout=dropout, batch_first=True))
            # self.linears.append(nn.Linear(n_hidden, nproj))
            # # self.linears.append(nn.Linear(2*n_hidden, 2*n_hidden))
    
    # def forward(self, feature):
        # recurrent, _ = self.rnns[0](feature)
        # output = self.linears[0](recurrent)

        # for i in range(self.num_layers-1):
            # output, _ = self.rnns[i+1](output)
            # output = self.linears[i+1](output)
        
        # return output

# class BLSTMP(nn.Module):
    # def __init__(self, n_in, n_hidden, nproj, rproj=100, dropout=0, num_layers=1):
        # super(BLSTMP, self).__init__()

        # self.num_layers = num_layers

        # self.rnns = nn.ModuleList([nn.LSTM(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True)])
        # self.linears = nn.ModuleList([nn.Linear(2*n_hidden, nproj)])

        # for i in range(num_layers-1):
            # self.rnns.append(nn.LSTM(nproj, n_hidden, bidirectional=True, dropout=dropout, batch_first=True))
            # self.linears.append(nn.Linear(2*n_hidden, nproj))
            # # self.linears.append(nn.Linear(2*n_hidden, 2*n_hidden))
    
    # def forward(self, feature):
        # recurrent, _ = self.rnns[0](feature)
        # output = self.linears[0](recurrent)

        # for i in range(self.num_layers-1):
            # output, _ = self.rnns[i+1](output)
            # output = self.linears[i+1](output)
        
        # return output

class TSVAD_module(NeuralModule, Exportable):
    """
    TS-VAD for overlap-aware diarization.
    Args:
        feat_in (int): size of the input features
        num_speakers(int): the number of speakers
    """

    @property
    def output_types(self):
        return OrderedDict({"probs": NeuralType(('B', 'T', 'C'), ProbsType())})
    
    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return OrderedDict(
            {
            "ms_embs": NeuralType(('B', 'T', 'C', 'D'), SpectrogramType()),
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
            feat_in: int,
            frame_splicing: int = 1,
            out_channels: list = [],
            num_spks: int = -1,
            scale_n: int = 1,
            lstm_hidden_size=512,
            cnn_output_ch=16,
            emb_sizes=192,
            dropout_rate: float=0.5):
        super().__init__()

        feat_in = feat_in * frame_splicing
        self._feat_in = feat_in
        self._num_classes = num_spks
        self.emb_sizes = 192
        self.num_spks = num_spks
        self.scale_n = scale_n
        self.chan = 1
        self.eps = 1e-6
        self.num_lstm_layers = 1
        self.fixed_ms_weight = torch.ones(self.scale_n)/self.scale_n
        self.cos_dist = torch.nn.CosineSimilarity(dim=3, eps=self.eps)
        lstm_input_size = lstm_hidden_size
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, num_layers=self.num_lstm_layers, batch_first=True)
        self.cnn = ConvLayer(in_channels=1, out_channels=cnn_output_ch, kernel_size=(num_spks+1, 1), stride=(0, 1))
        self.cnn2linear = nn.Linear(emb_sizes*cnn_output_ch, lstm_hidden_size)
        self.linear2weights = nn.Linear(lstm_hidden_size, self.scale_n)
        self.hidden2spk = nn.Linear(lstm_hidden_size, self.num_spks)
        self.dis2emb= nn.Linear(self.num_spks, lstm_input_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.hidden2spk.apply(self.init_weights)
        self.dis2emb.apply(self.init_weights)
        self.lstm.apply(self.init_weights)
    
    def core_model(self, ms_embs, length, ms_avg_embs, targets):
        """
        targets: batch_size x feats_len x max_spks
        avg_embs: batch_size x scale_n x emb_dim x max_spks
        ms_embs : batch_size x feats_len x scale_n x emb_dim
        """
        batch_size = ms_embs.shape[0]
        length = ms_embs.shape[1]
        # Extend ms_embs to have "max_spks" number of embeddings in spk axis
        sprint("ms_embs shape:", ms_embs.shape)
        _ms_embs = ms_embs.unsqueeze(4).expand(-1, -1, -1, -1, self.num_spks)
        
        # Extend ms_avg_embs to have "length" number of embeddings in temporal axis
        sprint("_ms_embs shape:", _ms_embs.shape)
        sprint("ms_avg_embs shape:", ms_avg_embs.shape)
        _ms_avg_embs = ms_avg_embs.unsqueeze(1).expand(-1, length, -1, -1, -1)

        # Cosine distance: batch_size x length x emb_dim x max_spks
        sprint("_ms_avg_embs shape:", _ms_avg_embs.shape)
        cos_dist_seq = self.cos_dist(_ms_embs, _ms_avg_embs)

        # Cosine weight: batch_size x length x emb_dim x max_spks
        sprint("cos_dist_seq shape:", cos_dist_seq.shape)
        sprint("self.fixed_ms_weight shape:", self.fixed_ms_weight.shape)
        self.fixed_ms_weight = self.fixed_ms_weight.to(ms_embs.device)
        cos_weight = self.fixed_ms_weight[None, None, :, None].expand(batch_size, length, -1, self.num_spks)

        # Element wise multiplied cosine dist vals: batch_size x length x emb_dim x max_spks
        sprint("cos_weight shape:", cos_weight.shape)
        weighted_cos_dist_seq = torch.mul(cos_weight, cos_dist_seq)

        # Calculate the weighted sum on the multi-scale axis: batch_size x length
        sprint("weighted_cos_dist_seq shape:", weighted_cos_dist_seq.shape)
        seq_input = weighted_cos_dist_seq.sum(axis=2)

        # Feed seq_input to the sequence modeler.    
        sprint("seq_input shape:", seq_input.shape)
        lstm_input = self.dis2emb(seq_input)
        lstm_input = F.relu(lstm_input)
        lstm_input = self.dropout(lstm_input)

        sprint("lstm_input shape:", lstm_input.shape)
        lstm_output, (hn, cn)= self.lstm(lstm_input)
        lstm_hidden_out = F.relu(lstm_output)
        lstm_hidden_out = self.dropout(lstm_hidden_out)

        sprint("lstm_hidden shape:", lstm_hidden_out.shape)
        spk_preds = self.hidden2spk(lstm_hidden_out)

        sprint("spk_preds shape:", spk_preds.shape)
        preds = nn.Sigmoid()(spk_preds)
        
        sprint("preds shape:", preds.shape)
        # import ipdb; ipdb.set_trace()
        return preds

    @typecheck()
    def forward(self, ms_embs, length, ms_avg_embs, targets):
        sprint("ms_embs 0:", ms_embs.shape)
        # bs, cnn_chan, tframe, dim = cnn_encoded.size()
        preds = self.core_model(ms_embs, length, ms_avg_embs, targets)
        return preds
    # def forward(self, ms_embs, length, ms_avg_embs):
        # sprint("ms_embs 0:", ms_embs.shape)
        # # bs, cnn_chan, tframe, dim = cnn_encoded.size()
        # preds = self.core_model(ms_embs, length, ms_avg_embs)
        # return preds

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
        # return tuple([input_example, lens, avg_embs])
        ### temp
        targets = torch.randn(1, lens, self.num_spks).round().float()
        return tuple([input_example, lens, avg_embs, targets])

