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

"""ConvSubsampling with per-stage time strides, frequency pooling and a projection bias flag."""

import pytest
import torch
from omegaconf import ListConfig

from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling, _layer_padding


def _strided(**overrides):
    options = dict(
        subsampling="striding",
        subsampling_factor=6,
        feat_in=8,
        feat_out=16,
        conv_channels=[2, 3, 3],
        conv_strides=[1, 3, 2],
        freq_pools=[2, 1, 1],
        out_bias=False,
    )
    options.update(overrides)
    return ConvSubsampling(**options).eval()


class TestConvSubsamplingStrides:
    @pytest.mark.unit
    def test_lengths_shapes_and_padding_invariance(self):
        torch.manual_seed(0)
        module = _strided()
        features = torch.randn(2, 50, 8)
        lengths = torch.tensor([50, 31])
        with torch.no_grad():
            out, out_lengths = module(x=features, lengths=lengths)
            alone, alone_lengths = module(x=features[1:, :31], lengths=lengths[1:])
        # time strides 1, 3, 2 with kernel 3 and padding 1: ceil(ceil(T / 3) / 2)
        assert out.shape == (2, 9, 16)
        assert out_lengths.tolist() == [9, 6] and alone_lengths.tolist() == [6]
        assert module.out.bias is None
        # the frequency axis is halved once by the pool, so the projection sees 3 channels x 4 bins
        assert module.out.weight.shape == (16, 12)
        torch.testing.assert_close(out[1, :6], alone[0, :6], atol=1e-5, rtol=1e-5)

    @pytest.mark.unit
    def test_channels_as_omegaconf_list(self):
        module = _strided(conv_channels=ListConfig([2, 3, 3]), freq_pools=ListConfig([2, 1, 1]))
        assert module.out.weight.shape == (16, 12)

    @pytest.mark.unit
    def test_validation(self):
        with pytest.raises(ValueError, match="product of conv_strides"):
            ConvSubsampling("striding", 4, 8, 16, [2, 2], conv_strides=[1, 3])
        with pytest.raises(ValueError, match="only supported with subsampling='striding'"):
            ConvSubsampling("dw_striding", 6, 8, 16, [2, 2, 2], conv_strides=[1, 3, 2])
        with pytest.raises(ValueError, match="one entry per stride"):
            ConvSubsampling("striding", 6, 8, 16, [2, 2], conv_strides=[1, 3, 2])

    @pytest.mark.unit
    def test_defaults_unchanged(self):
        module = ConvSubsampling("striding", 4, 8, 16, 4)
        assert module.out.bias is not None and module.out.weight.shape == (16, 8)
        features = torch.randn(1, 20, 8)
        out, out_lengths = module(x=features, lengths=torch.tensor([20]))
        assert out.shape == (1, 5, 16) and out_lengths.tolist() == [5]

    @pytest.mark.unit
    def test_layer_padding_of_pooling_layers(self):
        assert _layer_padding(torch.nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))) == (0, 0)
        assert _layer_padding(torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)) == (1, 1)
