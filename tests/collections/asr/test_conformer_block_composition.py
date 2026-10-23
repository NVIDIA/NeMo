# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Configurable blocks of the Conformer encoder: subsampling, feed-forward, convolution, attention and norm."""

import functools

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.modules import ConformerEncoder
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerConvolution, ConformerFeedForward
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling


def _composed_encoder(**overrides):
    options = dict(
        feat_in=8,
        n_layers=2,
        d_model=16,
        n_heads=2,
        subsampling_factor=8,  # overridden by the block's own factor
        xscaling=False,
        dropout=0.0,
        dropout_pre_encoder=0.0,
        dropout_emb=0.0,
        pre_encode=functools.partial(
            ConvSubsampling, subsampling="dw_striding", subsampling_factor=4, conv_channels=4
        ),
        feed_forward=functools.partial(ConformerFeedForward, activation="squared_relu", use_bias=False),
        conv=functools.partial(ConformerConvolution, kernel_size=4, norm_eps=1e-3),
        norm=functools.partial(torch.nn.LayerNorm, eps=1e-6),
    )
    options.update(overrides)
    return ConformerEncoder(**options).eval()


class TestConformerBlockComposition:
    @pytest.mark.unit
    def test_blocks_from_partials(self):
        encoder = _composed_encoder()
        assert isinstance(encoder.pre_encode, ConvSubsampling) and encoder.subsampling_factor == 4
        assert encoder.pre_encode._feat_in == 8 and encoder.pre_encode._feat_out == 16
        layer = encoder.layers[0]
        assert layer.feed_forward1.linear1.bias is None and layer.feed_forward2.linear2.bias is None
        assert type(layer.feed_forward1.activation).__name__ == "SquaredReLU"
        assert layer.conv.depthwise_conv.kernel_size == (4,)
        assert (layer.conv.depthwise_conv._left_padding, layer.conv.depthwise_conv._right_padding) == (1, 2)
        assert layer.conv.batch_norm.eps == 1e-3
        assert layer.norm_out.eps == 1e-6 and layer.norm_feed_forward1.eps == 1e-6 and layer.norm_conv.eps == 1e-6
        # the encoder follows the conv block's kernel for its streaming context
        assert encoder.conv_kernel_size == 4 and list(encoder.conv_context_size) == [1, 2]
        # attention keeps the bias: use_bias is the layer's default, the blocks only change their own arguments
        assert layer.self_attn.linear_q.bias is not None

    @pytest.mark.unit
    def test_even_kernel_padding_invariance(self):
        torch.manual_seed(0)
        encoder = _composed_encoder()
        features = torch.randn(2, 8, 40)
        lengths = torch.tensor([40, 25])
        with torch.no_grad():
            states, out_lengths = encoder(audio_signal=features, length=lengths)
            alone, alone_lengths = encoder(audio_signal=features[1:, :, :25], length=lengths[1:])
        assert states.shape == (2, 16, 10) and out_lengths.tolist() == [10, 7] and alone_lengths.tolist() == [7]
        torch.testing.assert_close(states[1, :, :7], alone[0, :, :7], atol=1e-5, rtol=1e-5)

    @pytest.mark.unit
    def test_blocks_from_hydra_sections(self):
        config = DictConfig(
            {
                "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                "feat_in": 8,
                "n_layers": 1,
                "d_model": 16,
                "n_heads": 2,
                "subsampling_conv_channels": 4,
                "feed_forward": {
                    "_target_": "nemo.collections.asr.parts.submodules.conformer_modules.ConformerFeedForward",
                    "_partial_": True,
                    "activation": "gelu",
                },
                "conv": {
                    "_target_": "nemo.collections.asr.parts.submodules.conformer_modules.ConformerConvolution",
                    "_partial_": True,
                    "kernel_size": 8,
                    "norm_type": "layer_norm",
                },
                "norm": {"_target_": "torch.nn.LayerNorm", "_partial_": True, "eps": 1e-6},
            }
        )
        encoder = ConformerEncoder.from_config_dict(config)
        layer = encoder.layers[0]
        assert isinstance(layer.feed_forward1.activation, torch.nn.GELU)
        assert layer.conv.depthwise_conv.kernel_size == (8,) and isinstance(layer.conv.batch_norm, torch.nn.LayerNorm)
        assert (layer.conv.depthwise_conv._left_padding, layer.conv.depthwise_conv._right_padding) == (3, 4)
        assert layer.norm_out.eps == 1e-6 and list(encoder.conv_context_size) == [3, 4]
        # the section's partial survives the round trip through the module's config
        assert encoder.to_config_dict()["conv"]["kernel_size"] == 8

    @pytest.mark.unit
    def test_attention_block(self):
        from nemo.collections.asr.parts.submodules.multi_head_attention import RelPositionMultiHeadAttention

        encoder = _composed_encoder(self_attn=functools.partial(RelPositionMultiHeadAttention, use_bias=False))
        assert encoder.layers[0].self_attn.linear_q.bias is None
        assert encoder.layers[0].self_attn.h == 2

    @pytest.mark.unit
    def test_defaults_unchanged(self):
        encoder = ConformerEncoder(feat_in=8, n_layers=1, d_model=16, n_heads=2, subsampling_conv_channels=4)
        layer = encoder.layers[0]
        assert type(layer.feed_forward1).__name__ == "ConformerFeedForward"
        assert type(layer.feed_forward1.activation).__name__ == "Swish"
        assert layer.feed_forward1.linear1.bias is not None
        assert type(layer.norm_out) is torch.nn.LayerNorm and layer.norm_out.eps == 1e-5
        assert layer.conv.batch_norm.eps == 1e-5
        assert (layer.conv.depthwise_conv._left_padding, layer.conv.depthwise_conv._right_padding) == (15, 15)
        assert list(encoder.conv_context_size) == [15, 15]
        assert encoder.pre_encode.out.bias is not None and encoder.subsampling_factor == 4
        # a standalone convolution module with an odd kernel keeps symmetric padding
        conv = ConformerConvolution(d_model=16, kernel_size=5)
        assert (conv.depthwise_conv._left_padding, conv.depthwise_conv._right_padding) == (2, 2)
