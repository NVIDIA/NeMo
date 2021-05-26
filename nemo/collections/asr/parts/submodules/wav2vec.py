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

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Tuple

import torch
from torch import nn

from nemo.collections.asr.models.wav2vec.wav2vec_config import Wav2VecConvExtractorMode, Wav2VecTransformerConfig


class TransposeLast(torch.nn.Module):
    """
    Transposes last dimension. Useful for adding to a sequential block.
    """

    def forward(self, x):
        return x.transpose(-2, -1)


class SamePad(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = kernel_size % 2 == 0

    def forward(self, x):
        if self.remove:
            x = x[:, :, :-1]
        return x


class ConvFeatureEncoder(nn.Module):
    """
        Converts input raw audio into features for downstream transformer model.
        Uses 1D convolutional blocks with GeLU activation.
    """

    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        mode: Wav2VecConvExtractorMode = Wav2VecConvExtractorMode.default,
        conv_bias: bool = False,
    ):
        super().__init__()

        def block(
            n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (is_layer_norm and is_group_norm) is False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Sequential(TransposeLast(), nn.LayerNorm(dim, elementwise_affine=True), TransposeLast()),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(make_conv(), nn.GroupNorm(dim, dim, affine=True), nn.GELU(),)
            else:
                return nn.Sequential(make_conv(), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode is Wav2VecConvExtractorMode.layer_norm,
                    is_group_norm=mode is Wav2VecConvExtractorMode.default and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


class Wav2VecTransformerEncoder(nn.Module):
    def __init__(self, cfg: Wav2VecTransformerConfig):
        super().__init__()

        conv_cfg = cfg.conv

        self.dropout = cfg.dropout
        self.embedding_dim = cfg.encoder.embedding_dim
        self.layer_norm_first = cfg.encoder.layer_norm_first

        # positional convolutional embeddings
        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=conv_cfg.conv_pos,
            padding=conv_cfg.conv_pos // 2,
            groups=conv_cfg.conv_pos_groups,
        )

        self.feature_dropout = nn.Dropout(self.dropout)

        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (conv_cfg.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(conv_cfg.conv_pos), nn.GELU())

        encoder_cfg = cfg.encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=encoder_cfg.num_attention_heads,
                dim_feedforward=encoder_cfg.ffn_embedding_dim,
                dropout=self.dropout,
                activation=encoder_cfg.activation_fn.value,
            ),
            num_layers=encoder_cfg.encoder_layers,
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None):
        x = self.extract_features(x, padding_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(self, x, padding_mask=None):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x += x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.feature_dropout(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bias will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, nn.TransformerEncoderLayer):
        module.self_attn.in_proj_weight.data.normal_(mean=0.0, std=0.02)
