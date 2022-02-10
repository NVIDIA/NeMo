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
    def __init__(self, in_channels=1, out_channels=2, kernel_size=5, stride=(1, 1), padding=1):
        super(ConvLayer, self).__init__()
        pad_size = (kernel_size-1)//2
        padding = (pad_size, pad_size)
        self.cnn = nn.Sequential(
                      nn.Conv2d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=kernel_size, 
                                stride=stride, 
                                padding=padding),
                      nn.ReLU(),
                      nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
                   )

    def forward(self, feature):
        feature = self.cnn(feature)
        return feature
     
class LSTMP(nn.Module):
    def __init__(self, n_in, n_hidden, nproj, rproj=100, dropout=0, num_layers=1):
        super(LSTMP, self).__init__()

        self.num_layers = num_layers

        self.rnns = nn.ModuleList([nn.LSTM(n_in, n_hidden, bidirectional=False, dropout=dropout, batch_first=True)])
        self.linears = nn.ModuleList([nn.Linear(n_hidden, nproj)])

        for i in range(num_layers-1):
            self.rnns.append(nn.LSTM(nproj, n_hidden, bidirectional=False, dropout=dropout, batch_first=True))
            self.linears.append(nn.Linear(n_hidden, nproj))
            # self.linears.append(nn.Linear(2*n_hidden, 2*n_hidden))
    
    def forward(self, feature):
        recurrent, _ = self.rnns[0](feature)
        output = self.linears[0](recurrent)

        for i in range(self.num_layers-1):
            output, _ = self.rnns[i+1](output)
            output = self.linears[i+1](output)
        
        return output

class BLSTMP(nn.Module):
    def __init__(self, n_in, n_hidden, nproj, rproj=100, dropout=0, num_layers=1):
        super(BLSTMP, self).__init__()

        self.num_layers = num_layers

        self.rnns = nn.ModuleList([nn.LSTM(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True)])
        self.linears = nn.ModuleList([nn.Linear(2*n_hidden, nproj)])

        for i in range(num_layers-1):
            self.rnns.append(nn.LSTM(nproj, n_hidden, bidirectional=True, dropout=dropout, batch_first=True))
            self.linears.append(nn.Linear(2*n_hidden, nproj))
            # self.linears.append(nn.Linear(2*n_hidden, 2*n_hidden))
    
    def forward(self, feature):
        recurrent, _ = self.rnns[0](feature)
        output = self.linears[0](recurrent)

        for i in range(self.num_layers-1):
            output, _ = self.rnns[i+1](output)
            output = self.linears[i+1](output)
        
        return output

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
            "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
            "ivectors": NeuralType(('B', 'D', 'C'), EncodedRepresentation()),
            }
        )
    # init_mode: Optional[str] = 'xavier_uniform',
    # out_channels=[64, 64, 128, 128], 
    def __init__(
            self, 
            feat_in: int,
            frame_splicing: int = 1,
            out_channels: list = [],
            num_spks: int = -1,
            lstm_hidden_size=1024,
            emb_sizes=192,
            y_stride=10,
            rproj=100, 
            nproj=100, 
            cell=100):
        super().__init__()

        feat_in = feat_in * frame_splicing
        self._feat_in = feat_in
        self._num_classes = num_spks
        self.emb_sizes = 192
        self.num_spks = num_spks
        self.chan = 1
        # self.input_types = OrderedDict()

        batchnorm = nn.BatchNorm2d(1, eps=0.001, momentum=0.99)
        
        conv_layer_1 = ConvLayer(in_channels=1, out_channels=out_channels[0])
        conv_layer_2 = ConvLayer(in_channels=out_channels[0], out_channels=out_channels[1])
        conv_layer_3 = ConvLayer(in_channels=out_channels[1], out_channels=out_channels[2], stride=(1, y_stride))
        conv_layer_4 = ConvLayer(in_channels=out_channels[2], out_channels=out_channels[3])
        
        self.cnn_encoder = nn.Sequential(
                      batchnorm,
                      conv_layer_1,
                      conv_layer_2,
                      conv_layer_3,
                      conv_layer_4
                   )
        cell_lin = 100 
        feat_lin_layer_1 = nn.Linear(feat_in, cell_lin)
        feat_lin_layer_2 = nn.Linear(cell_lin, cell_lin)
        feat_lin_layer_3 = nn.Linear(cell_lin, cell_lin)
        feat_lin_layer_4 = nn.Linear(cell_lin, feat_in)
        
        self.dnn_encoder = nn.Sequential(
                      batchnorm,
                      feat_lin_layer_1,
                      feat_lin_layer_2,
                      feat_lin_layer_3,
                      feat_lin_layer_4,
                   )
        # dim = feat_in
        combined_size = out_channels[-1]*feat_in + self.emb_sizes # If CNN
        # combined_size = 1*feat_in + self.emb_sizes # If DNN
        rnn_in = num_spks * 2*nproj
        # combine_in = (out_channels[-1]*feat_in + self.emb_sizes) 
        combine_in = combined_size * num_spks
        self.SD_linear = nn.Linear(combined_size, nproj)
        self.rnn_speaker_detection = BLSTMP(n_in=nproj, n_hidden=cell, nproj=2*nproj, num_layers=1)
        self.rnn_combine = BLSTMP(rnn_in, n_hidden=cell, nproj=2*nproj)
        self.output_layer = nn.Linear(2*nproj//num_spks, 1)
        self._feat_in = feat_in

        self.linear_layer = torch.nn.Linear(in_features=lstm_hidden_size, out_features=self._num_classes)
        # self.apply(lambda x: init_weights(x, mode=init_mode))

    def core_model(self, cnn_encoded, signal_lengths, ivectors):
        # bs = cnn_encoded.shape[0]
        # bs, cnn_chan, tframe, dim = cnn_encoded.shape
        bs, cnn_chan, dim, tframe = cnn_encoded.shape
        sprint(f"cnn_encoded input shape: bs, cnn_chan, dim, tframe, {bs, cnn_chan, dim, tframe}")
        # tframe = torch.max(signal_lengths).item()
        sprint("cnn_encoded shape0:", cnn_encoded.shape)
        cnn_encoded    = cnn_encoded.permute(0, 2, 1, 3)  # B x 1 x T x cnn_feat -> B x T x 1 x cnn_feat
        sprint("cnn_encoded shape1:", cnn_encoded.shape)
        cnn_encoded    = cnn_encoded.contiguous().view(bs, tframe, cnn_chan * dim)      # B x T x 2560
        sprint("cnn_encoded shape2:", cnn_encoded.shape)
        # cnn_encoded    = cnn_encoded.unsqueeze(1).repeat(1, self.num_spks, 1, 1)            # B x num_spks x T x 2560
        cnn_encoded    = cnn_encoded.unsqueeze(1).expand(-1, self.num_spks, -1, -1)            # B x num_spks x T x 2560
        sprint("cnn_encoded repeated shape3:", cnn_encoded.shape)
        
        ivectors = ivectors.view(bs, self.num_spks, self.emb_sizes).unsqueeze(2)  # B x num_spks x 1 x emb_sizes
        sprint("ivectors shape1:", ivectors.shape)
        # ivectors = ivectors.repeat(1, 1, tframe, 1)                             # B x num_spks x T x emb_sizes
        ivectors = ivectors.expand(-1, -1, tframe, -1)                             # B x num_spks x T x emb_sizes
        sprint("ivectors repeated shape2:", ivectors.shape)
        
        sd_in  = torch.cat((cnn_encoded, ivectors), dim=-1)                 # B x num_spks x T x 2660
        sprint("sd_in shape1:", sd_in.shape)
        sd_in  = self.SD_linear(sd_in).view(self.num_spks*bs, tframe, -1)   # num_spks * B x T x 384
        sprint("sd_in shape2:", sd_in.shape)
        
        sd_out = self.rnn_speaker_detection(sd_in)                           #  num_spks*B x T x 320
        sprint("sd_out shape0:", sd_out.shape)
        sd_out = sd_out.contiguous().view(bs, self.num_spks, tframe, -1)     #  B x num_spks x T x 320
        sd_out = sd_out.view(bs, self.num_spks, tframe, -1)     #  B x num_spks x T x 320
        sprint("sd_out shape1:", sd_out.shape)
        sd_out = sd_out.permute(0, 2, 1, 3)                                  #  B x T x num_spks x 320
        sprint("sd_out shape2:", sd_out.shape)
        sd_out = sd_out.contiguous().view(bs, tframe, -1)                    #  B x T x num_spks*320
        sd_out = sd_out.view(bs, tframe, -1)                    #  B x T x num_spks*320
        sprint("sd_out shape3:", sd_out.shape)

        outputs = self.rnn_combine(sd_out)                                   #  B x T x 320
        sprint("outputs shape0:", outputs.shape)
        outputs = outputs.contiguous().view(bs, tframe, self.num_spks, -1)   #  B x T x num_spks x 320/num_spks
        sprint("outputs shape1:", outputs.shape)
        preds   = self.output_layer(outputs).squeeze(-1)                     #  B x T x num_spks
        sprint("preds shape2:", preds.shape)
        preds   = nn.Sigmoid()(preds)
        sprint("preds shape3:", preds.shape)
        return preds        

    @typecheck()
    def forward(self, audio_signal, length, ivectors):
        sprint("audio_signal 0:", audio_signal.shape)
        audio_signal_single_ch = audio_signal.unsqueeze(1)
        cnn_encoded = self.cnn_encoder(audio_signal_single_ch)
        sprint("audio_signal 1:", audio_signal_single_ch.shape)
        
        # audio_signal_single_ch = audio_signal_single_ch.permute(0, 1, 3, 2)
        # dnn_encoded = self.dnn_encoder(audio_signal_single_ch)
        # dnn_encoded = dnn_encoded.permute(0,1,3,2) 
        # sprint("audio_signal 2:", dnn_encoded.shape)
        bs, cnn_chan, tframe, dim = cnn_encoded.size()
        preds = self.core_model(cnn_encoded, length, ivectors)
        return preds

    def input_example(self):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        device = next(self.parameters()).device
        input_example = torch.randn(1, self._feat_in, 8192, device=device)
        lens = torch.full(size=(input_example.shape[0],), fill_value=8192, device=device)
        ivectors = torch.randn(1, self.emb_sizes, self.num_spks, device=device)
        return tuple([input_example, lens, ivectors])

