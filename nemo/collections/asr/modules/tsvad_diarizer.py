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

__all__ = ['SampleConvASREncoder', 'LSTMDecoder', 'TSVAD']

class CNN_ReLU_BatchNorm(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=3, stride=(1, 1), padding=1):
        super(CNN_ReLU_BatchNorm, self).__init__()
        self.cnn = nn.Sequential(
                      nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                      nn.ReLU(),
                      nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
                   )

    def forward(self, feature):
        feature = self.cnn(feature)
        return feature
     

class BLSTMP(nn.Module):
    def __init__(self, n_in, n_hidden, nproj=160, dropout=0, num_layers=1):
        super(BLSTMP, self).__init__()

        self.num_layers = num_layers

        self.rnns = nn.ModuleList([nn.LSTM(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True)])
        self.linears = nn.ModuleList([nn.Linear(2*n_hidden, 2*nproj)])

        for i in range(num_layers-1):
            self.rnns.append(nn.LSTM(2*nproj, n_hidden, bidirectional=True, dropout=dropout, batch_first=True))
            self.linears.append(nn.Linear(2*n_hidden, 2*nproj))
    
    def forward(self, feature):
        recurrent, _ = self.rnns[0](feature)
        output = self.linears[0](recurrent)

        for i in range(self.num_layers-1):
            output, _ = self.rnns[i+1](output)
            output = self.linears[i+1](output)
        
        return output

class TSVAD(NeuralModule, Exportable):
    """
    TS-VAD for overlap-aware diarization.
    Args:
        feat_in (int): size of the input features
        num_speakers(int): the number of speakers
    """
    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types(self):
        return OrderedDict({"probs": NeuralType(('B', 'T', 'D'), ProbsType())})

    # @property
    # def input_types(self):
        # return OrderedDict({"encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation())})


    # def __init__(self, feat_in, num_speakers):
        # super().__init__()

        # if vocabulary is not None:
            # if num_classes != len(vocabulary):
                # raise ValueError(
                    # f"If vocabulary is specified, it's length should be equal to the num_classes. "
                    # f"Instead got: num_classes={num_classes} and len(vocabulary)={len(vocabulary)}"
                # )
            # self.__vocabulary = vocabulary
        
    def __init__(
            self, 
            feat_in: int,
            frame_splicing: int = 1,
            init_mode: Optional[str] = 'xavier_uniform',
            num_spks: int = 5,
            out_channels=[64, 64, 128, 128], 
            lstm_hidden_size=1024,
            emb_sizes=192,
            rproj=128, 
            nproj=160, 
            cell=896):
        super(TSVAD, self).__init__()

        feat_in = feat_in * frame_splicing
        self._feat_in = feat_in
        self._num_classes = num_spks
        self.emb_sizes = emb_sizes
        self.num_spks = 5
        self.chan = 1

        batchnorm = nn.BatchNorm2d(1, eps=0.001, momentum=0.99)
        
        cnn_relu_batchnorm1 = CNN_ReLU_BatchNorm(in_channels=1, out_channels=out_channels[0])
        cnn_relu_batchnorm2 = CNN_ReLU_BatchNorm(in_channels=out_channels[0], out_channels=out_channels[1])
        cnn_relu_batchnorm3 = CNN_ReLU_BatchNorm(in_channels=out_channels[1], out_channels=out_channels[2], stride=(1, 2))
        cnn_relu_batchnorm4 = CNN_ReLU_BatchNorm(in_channels=out_channels[2], out_channels=out_channels[3])
        
        self.cnn_encoder = nn.Sequential(
                      batchnorm,
                      cnn_relu_batchnorm1,
                      cnn_relu_batchnorm2,
                      cnn_relu_batchnorm3,
                      cnn_relu_batchnorm4
                   )
        
        self.linear = nn.Linear(out_channels[-1]*20+self.emb_sizes, 3*rproj)
        self.rnn_speaker_detection = BLSTMP(3*rproj, cell, num_layers=2)
        self.rnn_combine = BLSTMP(8*nproj, cell)

        self.output_layer = nn.Linear(nproj//2, 1)


        self._feat_in = feat_in

        self.linear_layer = torch.nn.Linear(in_features=lstm_hidden_size, out_features=self._num_classes)
        # self.apply(lambda x: init_weights(x, mode=init_mode))

    def core_model(self, cnn_encoded, signal_lengths, ivectors):
        bs = cnn_encoded.shape[0].item()
        tframe = torch.max(signal_lengths).item()
        cnn_encoded    = cnn_encoded.permute(0, 2, 1, 3)  # B x 1 x T x cnn_feat -> B x T x 1 x cnn_feat
        cnn_encoded    = cnn_encoded.contiguous().view(bs, taudio_signalframe, self.chan*self.dim)      # B x T x 2560
        cnn_encoded    = cnn_encoded.unsqueeze(1).repeat(1, self.num_spks, 1, 1)            # B x num_spks x T x 2560
        
        ivectors = ivectors.view(bs, self.num_spks, self.emb_sizes).unsqueeze(2)  # B x num_spks x 1 x emb_sizes
        ivectors = ivectors.repeat(1, 1, tframe, 1)                             # B x num_spks x T x emb_sizes
        
        sd_in  = torch.cat((cnn_encoded, ivectors), dim=-1)                  # B x num_spks x T x 2660
        sd_in  = self.linear(sd_in).view(self.num_spks*bs, tframe, -1)       # num_spks*B x T x 384
        
        sd_out = self.rnn_speaker_detection(sd_in)                           #  num_spks*B x T x 320
        sd_out = sd_out.contiguous().view(bs, self.num_spks, tframe, -1)     #  B x num_spks x T x 320
        sd_out = sd_out.permute(0, 2, 1, 3)                                  #  B x T x num_spks x 320
        sd_out = sd_out.contiguous().view(bs, tframe, -1)                    #  B x T x num_spks*320

        outputs = self.rnn_combine(sd_out)                                   #  B x T x 320
        outputs = outputs.contiguous().view(bs, tframe, self.num_spks, -1)   #  B x T x num_spks x 320/num_spks
        preds   = self.output_layer(outputs).squeeze(-1)                     #  B x T x num_spks
        preds   = nn.Sigmoid()(preds)
        return preds        


    @typecheck()
    def forward(self, feats, signal_lengths, ivectors):
        cnn_encoded = self.cnn_encoder(feats)
        bs, chan, tframe, dim = cnn_encoded.size()
        preds = self.core_model(cnn_encoded, signal_lengths, ivectors)
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
        return tuple([input_example, lens])

