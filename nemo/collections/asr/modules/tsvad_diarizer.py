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
     
class TSVAD_module(NeuralModule, Exportable):
    """
    TS-VAD for overlap-aware diarization.
    Args:
        feat_in (int): size of the input features
        num_speakers(int): the number of speakers
    """
        # scale_weights:  batch_size x length x scale_n

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
            dropout_rate: float=0.5,
            cnn_output_ch=4,
            emb_sizes=192,
            scale_n: int=1):
        super().__init__()

        self.emb_sizes = emb_sizes
        self.num_spks = num_spks
        self.scale_n = scale_n
        self.cnn_output_ch = cnn_output_ch
        self.chan = 1
        self.eps = 1e-6
        self.num_lstm_layers = 1
        self.fixed_ms_weight = torch.ones(self.scale_n)/self.scale_n
        self.softmax = torch.nn.Softmax(dim=2)
        self.cos_dist = torch.nn.CosineSimilarity(dim=3, eps=self.eps)
        lstm_input_size = hidden_size
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers=self.num_lstm_layers, batch_first=True)
        self.cnn = ConvLayer(in_channels=1, out_channels=cnn_output_ch, kernel_size=(self.scale_n + self.scale_n*num_spks, 1), stride=(1, 1))
        self.cnn2linear = nn.Linear(emb_sizes*cnn_output_ch, hidden_size)
        self.linear2weights = nn.Linear(hidden_size, self.scale_n)
        self.hidden2spk = nn.Linear(hidden_size, self.num_spks)

        ### Non-SUM mode
        self.dis2emb = nn.Linear(self.scale_n * self.num_spks, lstm_input_size)
        ### SUM mode
        # self.dis2emb = nn.Linear(self.num_spks, lstm_input_size)

        self.dropout = nn.Dropout(dropout_rate)

        self.hidden2spk.apply(self.init_weights)
        self.dis2emb.apply(self.init_weights)
        self.lstm.apply(self.init_weights)
    
    def core_model(self, ms_emb_seq, length, ms_avg_embs, targets):
        """
        targets: batch_size x feats_len x max_spks
        avg_embs: batch_size x scale_n x emb_dim x max_spks
        ms_emb_seq : batch_size x feats_len x scale_n x emb_dim
        """
        batch_size = ms_emb_seq.shape[0]
        length = ms_emb_seq.shape[1]
        emb_dim = ms_emb_seq.shape[-1]
        # Extend ms_emb_seq to have "max_spks" number of embeddings in spk axis
        sprint("ms_emb_seq shape:", ms_emb_seq.shape)
        
        # Stream 1
        _ms_emb_seq = ms_emb_seq.unsqueeze(4).expand(-1, -1, -1, -1, self.num_spks)
        # _ms_emb_seq: batch_size x feats_len x scale_n x emb_dim x num_spks
        
        # Stream 2: not expanding this with num_spks since this goes to CNN.
        _ms_emb_seq_single = ms_emb_seq
        # _ms_emb_seq_single: batch_size x feats_len x scale_n x emb_dim 


        # Extend ms_avg_embs to have "length" number of embeddings in temporal axis
        sprint("_ms_emb_seq shape:", _ms_emb_seq.shape)
        sprint("ms_avg_embs shape:", ms_avg_embs.shape)
        _ms_avg_embs = ms_avg_embs.unsqueeze(1).expand(-1, length, -1, -1, -1)
        # _ms_avg_embs: batch_size x feats_len x scale_n x emb_dim x max_spks

        # Stream1: Cosine distance: batch_size x feats_len x emb_dim x max_spks
        sprint("_ms_avg_embs shape:", _ms_avg_embs.shape)
        _ms_avg_embs_perm = _ms_avg_embs.permute(0, 1, 2, 4, 3).reshape(batch_size, length, -1, emb_dim)
        # _ms_avg_embs_perm: batch_size x feats_len x (scale_n * max_spks) x emb_dim 
        cos_dist_seq = self.cos_dist(_ms_emb_seq, _ms_avg_embs)
        sprint("cos_dist_seq shape:", cos_dist_seq.shape)
        # cos_dist_seq: batch_size x feats_len x scale_n x num_spks 

        # Stream2: CNN stream
        _ms_cnn_input_seq = torch.cat([_ms_avg_embs_perm, _ms_emb_seq_single], dim=2)
        # _ms_cnn_input_seq: batch_size x feats_len x (scale_n * max_spks + scale_n) x emb_dim 
        _ms_cnn_input_seq = _ms_cnn_input_seq.unsqueeze(2)
        # _ms_cnn_input_seq: batch_size x feats_len x 1 x (scale_n * max_spks + scale_n) x emb_dim 
        _ms_cnn_input_seq = _ms_cnn_input_seq.flatten(0, 1)
        cnn_out = self.cnn(_ms_cnn_input_seq)
        sprint("cnn_out shape:", cnn_out.shape)
        cnn_out = cnn_out.reshape(batch_size, length, self.cnn_output_ch, emb_dim)
        sprint("cnn_out reshape shape:", cnn_out.shape)
        # cnn_out = batch_size x cnn_output_ch x feats_len x emb_dim
        lin_input_seq = cnn_out.view(batch_size, length, self.cnn_output_ch * emb_dim)
        hidden_seq = self.cnn2linear(lin_input_seq)
        sprint("hidden_seq shape:", hidden_seq.shape)
        self.scale_weights = self.linear2weights(hidden_seq)
        self.scale_weights = self.softmax(self.scale_weights)
        sprint("self.scale_weights shape:", self.scale_weights.shape)
        # scale_weights:  batch_size x length x scale_n
        sprint("===MEAN self.scale_weights:", torch.mean(self.scale_weights, (0, 1)))
        

        # Cosine weight: batch_size x length x emb_dim x max_spks
        sprint("cos_dist_seq shape:", cos_dist_seq.shape)
        sprint("self.scale_weights shape:", self.scale_weights.shape)

        self.scale_weights = self.scale_weights.to(ms_emb_seq.device)
        # cos_weight = self.scale_weights[None, None, :, None].expand(batch_size, length, -1, self.num_spks)
        cos_weight = self.scale_weights.unsqueeze(3).expand(-1, -1, -1, self.num_spks)
        # cos_weight expanded:  batch_size x feats_len x scale_n x num_spks (should be the same as cos_dist_seq dim)

        # Element wise multiplied cosine dist vals: batch_size x length x emb_dim x max_spks
        sprint("cos_weight shape:", cos_weight.shape)
        weighted_cos_dist_seq = torch.mul(cos_weight, cos_dist_seq)

        # Calculate the weighted sum on the multi-scale axis: batch_size x length
        sprint("weighted_cos_dist_seq shape:", weighted_cos_dist_seq.shape)
        # seq_input = weighted_cos_dist_seq.sum(axis=2)

        # Feed seq_input to the sequence modeler.    
        # sprint("seq_input shape:", seq_input.shape)
        # lstm_input = self.dis2emb(seq_input)
        # sprint("After view weighted_cos_dist_seq shape:", weighted_cos_dist_seq.shape)
        weighted_cos_dist_seq = weighted_cos_dist_seq.view(batch_size, length, -1)
        lstm_input = self.dis2emb(weighted_cos_dist_seq)
        # sprint("seq_input shape:", weighted_cos_dist_seq.shape)
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
        return preds, self.scale_weights

    @typecheck()
    def forward(self, ms_emb_seq, length, ms_avg_embs, targets):
        sprint("ms_emb_seq 0:", ms_emb_seq.shape)
        # bs, cnn_chan, tframe, dim = cnn_encoded.size()
        # (preds, scale_weights) 
        preds, scale_weights = self.core_model(ms_emb_seq, length, ms_avg_embs, targets)
        return preds, scale_weights
    # def forward(self, ms_emb_seq, length, ms_avg_embs):
        # sprint("ms_emb_seq 0:", ms_emb_seq.shape)
        # # bs, cnn_chan, tframe, dim = cnn_encoded.size()
        # preds = self.core_model(ms_emb_seq, length, ms_avg_embs)
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

