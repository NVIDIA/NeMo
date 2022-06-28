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

import math
from collections import OrderedDict

import torch
import torch.distributed
import torch.nn as nn
from torch.nn import LayerNorm
from nemo.collections.asr.models.ss_model import EPS
import copy

from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule

__all__ = ['DualPathModel', 'DualBlock']

class DualBlock(NeuralModule, Exportable):
    """
    Computation block for dual-path processing

    Args:
        intra_model (NeuralModule) : model to process within chunks
        inter_model (NeuralModule) : model to process across chunks
        feat_out (int) : d_model of intra/inter
        skip_around_intra (bool) : skip connection 
        linear_layer_after_inter_intra (bool) : whether to use linear layer after intra/inter

    """
    def __init__(
        self,
        intra_model,
        inter_model,
        feat_out,
        skip_around_intra=True,
        linear_layer_after_inter_intra=False,
    ):
        super().__init__()
        self.intra_model = intra_model
        self.inter_model = inter_model
        self.skip_around_intra = skip_around_intra
        self.linear_layerafter_inter_intra = linear_layer_after_inter_intra

        # normalizations
        self.norm = 'layer_norm'
        self.intra_norm = nn.GroupNorm(1, feat_out, eps=EPS)
        self.inter_norm = nn.GroupNorm(1, feat_out, eps=EPS)

        # linear
        if linear_layer_after_inter_intra:
            self.intra_linear = nn.Linear(feat_out, feat_out)
            self.inter_linear = nn.Linear(feat_out, feat_out)

        
    def forward(self, x):
        """
        x : torch.Tensor
            [B, F, C, Nc]
            where, B = BatchSize, 
                F = feat size
                C = chunk length
                Nc = number of chunks
        """
        B, F, C, Nc = x.shape

        # intra model
        intra  = x.permute(0, 3, 2, 1).contiguous().view(B * Nc, C, F)
        # [B*Nc, C, F]
        
        intra = self.intra_model(intra)

        if self.linear_layerafter_inter_intra:
            intra = self.intra_linear(intra)

        intra = intra.view(B, Nc, C, F)
        # [B, Nc, C, F]

        intra = intra.permute(0, 3, 2, 1).contiguous()
        # [B, F, C, Nc]

        if self.norm is not None: 
            intra = self.intra_norm(intra)

        if self.skip_around_intra:
            intra = intra + x

        
        # inter model
        inter = intra.permute(0, 2, 3, 1).contiguous().view(B * C, Nc, F)
        # [B*C, Nc, F]

        inter = self.inter_model(inter)

        if self.linear_layerafter_inter_intra:
            inter = self.inter_linear(inter)

        inter = inter.view(B, C, Nc, F)
        # [B, C, Nc, F]

        inter = inter.permute(0, 3, 1, 2).contiguous()
        # [B, F, C, Nc]

        if self.norm is not None:
            inter = self.inter_norm(inter)

        # skip connection
        out = inter + intra

        return out

class DualPathModel(NeuralModule, Exportable):
    """
    implementation of dual path model for speech separation

    Args:
        num_speakers (int) : number of sources (speakers)
        feat_in (int) :  number of channels at the output of encoder
        feat_out (int) : number of channels at input of intra and inter blocks
        intra_model (dict) : parameters for intra model
        inter_model (dict)  : parameters for inter model
        num_layers (int) : number of layers of dual block (intra + inter)
        chunk_len (int) : chunk size
        linear_layer_after_inter_intra (bool) : whether to use linear inter and intra
        skip_around_intra (bool) : skip connection around intra
        max_seq_length (int) : maximum sequence length
    """



    def __init__(
        self,
        feat_in,
        feat_out,
        intra_model,
        inter_model,
        num_layers=1,
        num_speakers=2,
        chunk_len=250,
        skip_around_intra=True,
        linear_layer_after_inter_intra=False,
        max_seq_length=20000,
        *args,
        **kwargs,
    ):
        super().__init__()
        
        self.num_speakers = num_speakers
        self.chunk_len = chunk_len
        self.num_layers = num_layers

        self.norm = nn.GroupNorm(1, feat_in, eps=EPS)
        self.conv1d = nn.Conv1d(feat_in, feat_out, 1, bias=False)

        
        if intra_model.get('model_type', None) == 'transformer':
            intra_model = TransformerEncoder(
                n_layers=intra_model['num_layers'],
                d_model=intra_model['d_model'],
                ff_expansion_factor=intra_model['ff_expansion_factor'],
                self_attention_model=intra_model['pos_encoding'],
                n_heads=intra_model['n_heads'],
                pre_norm=intra_model['norm_before'],
            )
        else:
            raise ValueError(f"{intra_model.get('model_type')} is not valid for intra_model")

        if inter_model.get('model_type', None) == 'transformer':
            inter_model = TransformerEncoder(
                n_layers=inter_model['num_layers'],
                d_model=inter_model['d_model'],
                ff_expansion_factor=inter_model['ff_expansion_factor'],
                self_attention_model=inter_model['pos_encoding'],
                n_heads=inter_model['n_heads'],
                pre_norm=inter_model['norm_before'],
            )
        else:
            raise ValueError(f"{inter_model.get('model_type')} is not valid for inter_model")


        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            layer = copy.deepcopy(
                DualBlock(
                    intra_model=intra_model,
                    inter_model=inter_model,
                    feat_out=feat_out,
                    skip_around_intra=skip_around_intra,
                    linear_layer_after_inter_intra=linear_layer_after_inter_intra,
                )
            )
            self.layers.append(layer)

        
        self.conv2d = nn.Conv2d(feat_out, feat_out*num_speakers, kernel_size=1)
        self.end_conv1x1  = nn.Conv1d(feat_out, feat_in, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()

        # gated output layeddr
        self.output = nn.Sequential(
            nn.Conv1d(feat_out, feat_out, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(feat_out, feat_out, 1), nn.Sigmoid()
        )

    
    def forward(self, x):
        """
        Return output tensor

        Args:
            x : torch.Tensor
                [B, F, N]
                B: batch
                F: feat size
                N: seq length / num of time points

        Returns:
            out: torch.Tensor
                [spks, B, F, N]
        """
        # norm+ linear
        x = self.norm(x)    # make sure that norm is acting on correct dimension.
        x = self.conv1d(x)

        # chunk
        hop = self.chunk_len // 2
        x, pad_len = self._chunk(x, self.chunk_len, hop)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.prelu(x)
        # [B, F, C, Nc]

        x = self.conv2d(x)
        # [B, F*num_speakers, C, Nc]
        B, _, C, Nc = x.shape

        x = x.view(B * self.num_speakers, -1, C, Nc)
        # [B*num_speakers, F, C, Nc]

        x = self._overlap_add(x, pad_len, hop)
        x = self.output(x) * self.output_gate(x)

        x = self.end_conv1x1(x)
        # [B*num_speakers, F, N]

        _, F, N = x.shape
        x = x.view(B, self.num_speakers, F, N)
        x = self.activation(x)

        x = x.transpose(0, 1)
        # [num_speakers, B, F, N]

        return x 


    def _overlap_add(self, x, pad_len, hop):
        """
        Merge through overlap and add
        """

        B, F, C, Nc = x.shape

        x = x.transpose(2, 3).contiguous().view(B, F, -1, C * 2)

        x1 = x[:, :, :, :C].contiguous().view(B, F, -1)[:, :, hop:]
        x2 = x[:, :, :, C:].contiguous().view(B, F, -1)[:, :, :-hop]
        x = x1 + x2

        if pad_len > 0:
            x = x[:, :, :-pad_len]

        return x 


    def _chunk(self, x, chunk_len, hop):
        """
        Segment and stack encoder output
        
        Args:
            x: torch.Tensor
                [B, F, N]
                [B, N ,L]
            chunk_len: length of chunks
            hop : hop size

        Return:
            output: torch.Tensor
                [B, F, C, Nc]
            pad: padding used 
        """
        B, F, N = x.shape
        x, pad_len = self._padding(x, chunk_len, hop)
        x1 = x[:, :, :-hop].contiguous().view(B, F, -1, chunk_len)
        x2 = x[:, :, hop:].contiguous().view(B, F, -1, chunk_len)
        x = torch.cat([x1, x2], dim=3).view(B, F, -1, chunk_len).transpose(2,3)
        return x.contiguous(), pad_len


    def _padding(self, x, chunk_len, hop):
        """
        pad for whole number of chunks
        """
        B, F, N = x.shape
        pad_len = chunk_len  - (hop + N % chunk_len) % chunk_len
        if pad_len > 0:
            pad = torch.Tensor(torch.zeros(B, F, pad_len)).type(x.type())
            x = torch.cat([x, pad], dim=2)

        # for ease of overlap chunk
        _pad = torch.Tensor(torch.zeros(B, F, hop)).type(x.type())
        x = torch.cat([_pad, x, _pad], dim=2)

        return x, pad_len


