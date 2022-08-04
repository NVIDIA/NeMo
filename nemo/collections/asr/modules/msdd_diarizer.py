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
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    EncodedRepresentation,
    LengthsType,
    LogitsType,
    LogprobsType,
    NeuralType,
    SpectrogramType,
)
from nemo.core.neural_types.elements import ProbsType

__all__ = ['LSTMDecoder', 'MSDD_module']


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=(3, 1), stride=(1, 1)):
        super(ConvLayer, self).__init__()
        pad_size = (kernel_size[1] - 1) // 2
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
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
        num_spks (int):
            Max number of speakers that are processed by the model. In `MSDD_module`, `num_spks=2` for pairwise inference.
        hidden_size (int):
            Number of hidden units in sequence models and intermediate layers.
        num_lstm_layers (int):
            Number of the stacked LSTM layers.
        dropout_rate (float):
            Dropout rate for linear layers, CNN and LSTM.
        cnn_output_ch (int):
            Number of channels per each CNN layer.
        emb_sizes (int):
            Dimension of the embedding vectors.
        scale_n (int):
            Number of scales in multi-scale system.
        clamp_max (float):
            Maximum value for limiting the scale weight values.
        conv_repeat (int):
            Number of CNN layers after the first CNN layer.
        weighting_scheme (str):
            Name of the methods for estimating the scale weights.
        use_cos_sim_input (bool):
            If True, cosine similarity values are used for the input of the sequence models.
            If False, element-wise product values are used for the input of the sequence models.
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
        num_spks: int = 2,
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        dropout_rate: float = 0.5,
        cnn_output_ch: int = 16,
        emb_sizes: int = 192,
        scale_n: int = 1,
        clamp_max: float = 1.0,
        conv_repeat: int = 1,
        weighting_scheme: str = 'conv_scale_weight',
        use_cos_sim_input: bool = True,
    ):
        super().__init__()
        self._speaker_model = None
        self.batch_size: int = 1
        self.length: int = 50
        self.emb_sizes: int = emb_sizes
        self.num_spks: int = num_spks
        self.scale_n: int = scale_n
        self.cnn_output_ch: int = cnn_output_ch
        self.conv_repeat: int = conv_repeat
        self.chan: int = 2
        self.eps: float = 1e-6
        self.num_lstm_layers: int = num_lstm_layers
        self.weighting_scheme: str = weighting_scheme
        self.use_cos_sim_input: bool = use_cos_sim_input

        self.softmax = torch.nn.Softmax(dim=2)
        self.cos_dist = torch.nn.CosineSimilarity(dim=3, eps=self.eps)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate,
        )

        if self.weighting_scheme == 'conv_scale_weight':
            self.conv = nn.ModuleList(
                [
                    ConvLayer(
                        in_channels=1,
                        out_channels=cnn_output_ch,
                        kernel_size=(self.scale_n + self.scale_n * num_spks, 1),
                        stride=(1, 1),
                    )
                ]
            )
            for conv_idx in range(1, conv_repeat + 1):
                self.conv.append(
                    ConvLayer(
                        in_channels=1, out_channels=cnn_output_ch, kernel_size=(self.cnn_output_ch, 1), stride=(1, 1)
                    )
                )
            self.conv_bn = nn.ModuleList()
            for conv_idx in range(self.conv_repeat + 1):
                self.conv_bn.append(nn.BatchNorm2d(self.emb_sizes, affine=False))
            self.conv_to_linear = nn.Linear(emb_sizes * cnn_output_ch, hidden_size)
            self.linear_to_weights = nn.Linear(hidden_size, self.scale_n)

        elif self.weighting_scheme == 'attn_scale_weight':
            self.W_a = nn.Linear(emb_sizes, emb_sizes, bias=False)
            nn.init.eye_(self.W_a.weight)
        else:
            raise ValueError(f"No such weighting scheme as {self.weighting_scheme}")

        self.hidden_to_spks = nn.Linear(2 * hidden_size, self.num_spks)
        if self.use_cos_sim_input:
            self.dist_to_emb = nn.Linear(self.scale_n * self.num_spks, hidden_size)
            self.dist_to_emb.apply(self.init_weights)
        else:
            self.product_to_emb = nn.Linear(self.emb_sizes * self.num_spks, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_to_spks.apply(self.init_weights)
        self.lstm.apply(self.init_weights)
        self.clamp_max = clamp_max

    def core_model(self, ms_emb_seq, length, ms_avg_embs, targets):
        """
        Core model that accepts multi-scale cosine similarity values and estimates per-speaker binary label.

        Args:
            targets (Tensor):
                Ground-truth labels for the finest segment.
                Shape: (batch_size, feats_len, max_spks)
            avg_embs (Tensor):
                Cluster-average speaker embedding vectors.
                Shape: (batch_size, scale_n, self.emb_dim, max_spks)
            ms_emb_seq (Tensor):
                Multiscale input embedding sequence
                Shape: (batch_size, feats_len, scale_n, emb_dim)

        Returns:
            preds (Tensor):
                Predicted binary speaker label for each speaker.
                Shape: (batch_size, feats_len, max_spks)
            scale_weights (Tensor):
                Multiscale weights per each base-scale segment.
                Shape: (batch_size, feats_len, scale_n, max_spks)

        """
        self.batch_size = ms_emb_seq.shape[0]
        self.length = ms_emb_seq.shape[1]
        self.emb_dim = ms_emb_seq.shape[-1]

        _ms_emb_seq = ms_emb_seq.unsqueeze(4).expand(-1, -1, -1, -1, self.num_spks)
        ms_emb_seq_single = ms_emb_seq
        ms_avg_embs = ms_avg_embs.unsqueeze(1).expand(-1, self.length, -1, -1, -1)

        ms_avg_embs_perm = ms_avg_embs.permute(0, 1, 2, 4, 3).reshape(self.batch_size, self.length, -1, self.emb_dim)

        if self.weighting_scheme == "conv_scale_weight":
            scale_weights = self.conv_scale_weights(ms_avg_embs_perm, ms_emb_seq_single)
        elif self.weighting_scheme == "attn_scale_weight":
            scale_weights = self.attention_scale_weights(ms_avg_embs_perm, ms_emb_seq_single)
        else:
            raise ValueError(f"No such weighting scheme as {self.weighting_scheme}")
        scale_weights = scale_weights.to(ms_emb_seq.device)

        if self.use_cos_sim_input:
            context_emb = self.cosine_similarity(scale_weights, ms_avg_embs, _ms_emb_seq)
        else:
            context_emb = self.element_wise_product(scale_weights, ms_avg_embs, _ms_emb_seq)

        context_emb = self.dropout(F.relu(context_emb))
        lstm_output, (hn, cn) = self.lstm(context_emb)
        lstm_hidden_out = self.dropout(F.relu(lstm_output))
        spk_preds = self.hidden_to_spks(lstm_hidden_out)
        preds = nn.Sigmoid()(spk_preds)
        return preds, scale_weights

    def element_wise_product(self, scale_weights, ms_avg_embs, ms_emb_seq):
        """
        Calculate element wise product values among cluster-average embedding vectors and input embedding vector sequences.
        `element_wise_product` method usually takes more time to converge compared to `cosine_similarity` method.

        Args:
            scale_weights (Tensor):
                Multiscale weight vector.
                Shape: (batch_size, feats_len, scale_n, max_spks)
            ms_avg_embs_perm (Tensor):
                Tensor containing cluster-average speaker embeddings for each scale.
                Shape: (batch_size, length, scale_n, emb_dim)
            _ms_emb_seq (Tensor):
                Tensor containing multi-scale speaker embedding sequences. ms_emb_seq_single is input from the
                given audio stream input.
                Shape: (batch_size, length, num_spks, emb_dim)

        Returns:
            context_emb (Tensor):
                Output of `dist_to_emb` linear layer containing context for speaker label estimation.
        """
        scale_weight_flatten = scale_weights.reshape(self.batch_size * self.length, self.num_spks, self.scale_n)
        ms_avg_embs_flatten = ms_avg_embs.reshape(
            self.batch_size * self.length, self.scale_n, self.emb_dim, self.num_spks
        )
        ms_emb_seq_flatten = ms_emb_seq.reshape(-1, self.scale_n, self.emb_dim)
        ms_emb_seq_flatten_rep = ms_emb_seq_flatten.unsqueeze(3).reshape(
            -1, self.scale_n, self.emb_sizes, self.num_spks
        )
        elemwise_product = ms_avg_embs_flatten * ms_emb_seq_flatten_rep
        context_vectors = torch.bmm(
            scale_weight_flatten.reshape(self.batch_size * self.num_spks * self.length, 1, self.scale_n),
            elemwise_product.reshape(self.batch_size * self.num_spks * self.length, self.scale_n, self.emb_dim),
        )
        context_vectors = context_vectors.reshape(self.batch_size, self.length, self.emb_dim * self.num_spks)
        context_emb = self.product_to_emb(context_vectors)
        return context_emb

    def cosine_similarity(self, scale_weights, ms_avg_embs, _ms_emb_seq):
        """
        Calculate cosine similarity values among cluster-average embedding vectors and input embedding vector sequences.

        Args:
            scale_weights (Tensor):
                Multiscale weight vector.
                Shape: (batch_size, feats_len, scale_n, max_spks)
            ms_avg_embs_perm (Tensor):
                Tensor containing cluster-average speaker embeddings for each scale.
                Shape: (batch_size, length, scale_n, emb_dim)
            _ms_emb_seq (Tensor):
                Tensor containing multi-scale speaker embedding sequences. ms_emb_seq_single is input from the
                given audio stream input.
                Shape: (batch_size, length, num_spks, emb_dim)

        Returns:
            context_emb (Tensor):
                Output of `dist_to_emb` linear layer containing context for speaker label estimation.
        """
        cos_dist_seq = self.cos_dist(_ms_emb_seq, ms_avg_embs)
        context_vectors = torch.mul(scale_weights, cos_dist_seq)
        seq_input_sum = context_vectors.sum(axis=2).view(self.batch_size, self.length, -1)
        context_vectors = context_vectors.view(self.batch_size, self.length, -1)
        context_emb = self.dist_to_emb(context_vectors)
        return context_emb

    def attention_scale_weights(self, ms_avg_embs_perm, ms_emb_seq):
        """
        Use weighted inner product for calculating each scale weight. W_a matrix has (emb_dim * emb_dim) learnable parameters
        and W_a matrix is initialized with an identity matrix. Compared to "conv" method, this method shows more evenly
        distributed scale weights.

        Args:
            ms_avg_embs_perm (Tensor):
                Tensor containing cluster-average speaker embeddings for each scale.
                Shape: (batch_size, length, scale_n, emb_dim)
            ms_emb_seq_single (Tensor):
                Tensor containing multi-scale speaker embedding sequences. ms_emb_seq_single is input from the
                given audio stream input.
                Shape: (batch_size, length, num_spks, emb_dim)

        Returns:
            scale_weights (Tensor):
                Weight vectors that determines the weight of each scale.
                Shape: (batch_size, length, num_spks, emb_dim)
        """
        self.W_a(ms_emb_seq.flatten(0, 1))
        mat_a = self.W_a(ms_emb_seq.flatten(0, 1))
        mat_b = ms_avg_embs_perm.flatten(0, 1).permute(0, 2, 1)

        weighted_corr = torch.matmul(mat_a, mat_b).reshape(-1, self.scale_n, self.scale_n, self.num_spks)
        scale_weights = torch.sigmoid(torch.diagonal(weighted_corr, dim1=1, dim2=2))
        scale_weights = scale_weights.reshape(self.batch_size, self.length, self.scale_n, self.num_spks)
        scale_weights = self.softmax(scale_weights)
        return scale_weights

    def conv_scale_weights(self, ms_avg_embs_perm, ms_emb_seq_single):
        """
        Use multiple Convnet layers to estimate the scale weights based on the cluster-average embedding and
        input embedding sequence.

        Args:
            ms_avg_embs_perm (Tensor):
                Tensor containing cluster-average speaker embeddings for each scale.
                Shape: (batch_size, length, scale_n, emb_dim)
            ms_emb_seq_single (Tensor):
                Tensor containing multi-scale speaker embedding sequences. ms_emb_seq_single is input from the
                given audio stream input.
                Shape: (batch_size, length, num_spks, emb_dim)

        Returns:
            scale_weights (Tensor):
                Weight vectors that determines the weight of each scale.
                Shape: (batch_size, length, num_spks, emb_dim)
        """
        ms_cnn_input_seq = torch.cat([ms_avg_embs_perm, ms_emb_seq_single], dim=2)
        ms_cnn_input_seq = ms_cnn_input_seq.unsqueeze(2).flatten(0, 1)

        conv_out = self.conv_forward(
            ms_cnn_input_seq, conv_module=self.conv[0], bn_module=self.conv_bn[0], first_layer=True
        )
        for conv_idx in range(1, self.conv_repeat + 1):
            conv_out = self.conv_forward(
                conv_input=conv_out,
                conv_module=self.conv[conv_idx],
                bn_module=self.conv_bn[conv_idx],
                first_layer=False,
            )

        lin_input_seq = conv_out.view(self.batch_size, self.length, self.cnn_output_ch * self.emb_dim)
        hidden_seq = self.conv_to_linear(lin_input_seq)
        hidden_seq = self.dropout(F.leaky_relu(hidden_seq))
        scale_weights = self.softmax(self.linear_to_weights(hidden_seq))
        scale_weights = scale_weights.unsqueeze(3).expand(-1, -1, -1, self.num_spks)
        return scale_weights

    def conv_forward(self, conv_input, conv_module, bn_module, first_layer=False):
        """
        A module for Convolutional neural networks with 1-D filters. As a unit layer batch normalization, non-linear layer and dropout
        modules are included.

        Note:
            If `first_layer=True`, the input shape is set for processing embedding input.
            If `first_layer=False`, then the input shape is set for processing the output from another `conv_forward` module.

        Args:
            conv_input (Tensor):
                Reshaped tensor containing cluster-average embeddings and multi-scale embedding sequences.
                Shape: (batch_size*length, 1, scale_n*(num_spks+1), emb_dim)
            conv_module (ConvLayer):
                ConvLayer instance containing torch.nn.modules.conv modules.
            bn_module (torch.nn.modules.batchnorm.BatchNorm2d):
                Predefined Batchnorm module.
            first_layer (bool):
                Boolean for switching between first layer and the others.
                Default: `False`

        Returns:
            conv_out (Tensor):
                Convnet output that can be fed to another ConvLayer module or linear layer.
                Shape: (batch_size*length, 1, cnn_output_ch, emb_dim)
        """
        conv_out = conv_module(conv_input)
        conv_out = conv_out.permute(0, 2, 1, 3) if not first_layer else conv_out
        conv_out = conv_out.reshape(self.batch_size, self.length, self.cnn_output_ch, self.emb_dim)
        conv_out = conv_out.unsqueeze(2).flatten(0, 1)
        conv_out = bn_module(conv_out.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        conv_out = self.dropout(F.leaky_relu(conv_out))
        return conv_out

    @typecheck()
    def forward(self, ms_emb_seq, length, ms_avg_embs, targets):
        preds, scale_weights = self.core_model(ms_emb_seq, length, ms_avg_embs, targets)
        return preds, scale_weights

    def input_example(self):
        """
        Generate input examples for tracing etc.

        Returns (tuple):
            A tuple of input examples.
        """
        device = next(self.parameters()).device
        lens = torch.full(size=(input_example.shape[0],), fill_value=123, device=device)
        input_example = torch.randn(1, lens, self.scale_n, self.emb_sizes, device=device)
        avg_embs = torch.randn(1, self.scale_n, self.emb_sizes, self.num_spks, device=device)
        targets = torch.randn(1, lens, self.num_spks).round().float()
        return tuple([input_example, lens, avg_embs, targets])
