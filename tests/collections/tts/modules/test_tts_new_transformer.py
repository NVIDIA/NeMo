# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import random

import numpy as np
import pytest
import torch

from nemo.collections.tts.modules.transformer_2412 import (
    ConvolutionLayer,
    CrossAttention,
    PositionwiseConvFF,
    SelfAttention,
    Transformer,
    TransformerLayer,
)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# @pytest.mark.unit
# class TestConvolutionLayer:
#     @classmethod
#     def setup_class(cls):
#         cls.in_channels = 3
#         cls.out_channels = 6
#         cls.kernel_size = 3
#         cls.stride = 1
#         cls.dilation = 1
#         cls.bias = True
#
#     def test_non_causal_forward(self):
#         set_seed(1234)
#         layer = ConvolutionLayer(
#             self.in_channels,
#             self.out_channels,
#             self.kernel_size,
#             self.stride,
#             dilation=self.dilation,
#             bias=self.bias,
#             is_causal=False,
#         )
#
#         input_tensor = torch.randn(1, self.in_channels, 10)
#         output_tensor = layer(input_tensor)
#
#         # Mock expected output tensor
#         expected_output_tensor = torch.Tensor(
#             [[[ 0.4501,  0.0445,  0.3106,  0.9073,  0.0153,  0.4479,  0.4174, -0.3571,  0.3856,  0.3008],
#               [ 0.0426, -0.3726,  0.3657, -0.1988,  0.2904, -0.1108, -0.5632, 0.0358,  0.0257, -0.4434],
#               [-0.4830, -0.1834,  0.0454, -0.4382, -0.1952, -0.3125, -0.3263, -0.0384,  0.2328, -0.3073],
#               [ 0.2821, -0.3242, -0.5154, -0.2740, -0.5626,  0.1581, -0.1381, -0.8880, -0.1268, -0.0729],
#               [-0.8277, -0.3530, -0.4945, -0.0148,  0.7112,  0.1734,  0.5686, 1.0718,  0.3391,  0.5865],
#               [ 1.0760,  0.8330,  0.1915, -0.8854, -0.9524, -0.4001, -0.3806, -0.6032, -0.7552, -0.5738]]])
#
#         assert output_tensor.shape == expected_output_tensor.shape
#         assert torch.allclose(output_tensor.detach(), expected_output_tensor, atol=1e-6)
#
#     def test_causal_forward(self):
#         set_seed(1234)
#         layer = ConvolutionLayer(
#             self.in_channels,
#             self.out_channels,
#             self.kernel_size,
#             self.stride,
#             dilation=self.dilation,
#             bias=self.bias,
#             is_causal=True,
#         )
#         input_tensor = torch.randn(1, self.in_channels, 10)
#         output_tensor = layer(input_tensor)
#
#         # Mock expected output tensor
#         expected_output_tensor = torch.Tensor(
#             [[[-0.2415,  0.0911,  0.9798, -0.4081, -0.6563,  0.3625,  0.2462, 0.1406,  1.0330,  1.0457],
#               [-0.3430,  0.8194, -0.3609, -0.6095,  0.9169, -0.1487, -0.2949, -0.1173, -0.2240,  0.0783],
#               [-0.0992, -0.1366, -0.4898,  0.5433,  0.4675, -0.7876,  0.1261, -0.4229, -0.5769,  0.1363],
#               [-0.4830,  0.0078,  0.8823, -0.5039, -0.0125,  0.1992,  0.2243, -0.2240, -0.3468,  0.2797],
#               [ 0.2673, -0.3652, -0.7713,  0.6312, -0.7989, -0.1303, -0.4450, -0.2814, -0.0855, -0.3503],
#               [-0.2055, -0.1840,  0.4851, -0.0067,  0.9291, -0.0631,  0.2972, 0.1876, -1.0274, -0.8606]]]
#         )
#
#         assert output_tensor.shape == expected_output_tensor.shape
#         assert torch.allclose(output_tensor, expected_output_tensor, atol=1e-6)
#
#
# class TestPositionwiseConvFF:
#     @classmethod
#     def setup_class(cls):
#         cls.d_model = 3
#         cls.d_ffn = 12
#         cls.p_dropout = 0.1
#         cls.kernel_size = 3
#         cls.bias = True
#         cls.is_causal = False
#
#     def test_forward(self):
#         layer = PositionwiseConvFF(
#             self.d_model, self.d_ffn, self.p_dropout, self.kernel_size, bias=self.bias, is_causal=self.is_causal
#         )
#         input_tensor = torch.randn(1, 10, self.d_model)
#         output_tensor = layer(input_tensor)
#
#         # Mock expected output tensor
#         expected_output_tensor = torch.randn(1, 10, self.d_model)
#         torch.manual_seed(0)  # Ensure reproducibility
#         torch.nn.init.normal_(expected_output_tensor, mean=0.0, std=1.0)
#
#         assert output_tensor.shape == expected_output_tensor.shape
#         assert torch.allclose(output_tensor, expected_output_tensor, atol=1e-6)
#
#
# class TestSelfAttention:
#     @classmethod
#     def setup_class(cls):
#         cls.n_heads = 2
#         cls.d_model = 4
#         cls.p_dropout = 0.1
#         cls.is_causal = True
#         cls.max_length_causal_mask = 10
#
#     def test_forward(self):
#         layer = SelfAttention(
#             self.n_heads,
#             self.d_model,
#             self.p_dropout,
#             is_causal=self.is_causal,
#             max_length_causal_mask=self.max_length_causal_mask,
#         )
#         input_tensor = torch.randn(1, 10, self.d_model)
#         mask_tensor = torch.ones(1, 10).bool()
#         output_tensor, attn_prob = layer(input_tensor, mask_tensor)
#
#         # Mock expected output tensor
#         expected_output_tensor = torch.randn(1, 10, self.d_model)
#         expected_attn_prob = [torch.randn(1, self.n_heads, 10, 10), torch.randn(1, self.n_heads, 10, 10)]
#         torch.manual_seed(0)  # Ensure reproducibility
#         torch.nn.init.normal_(expected_output_tensor, mean=0.0, std=1.0)
#         for i in range(len(expected_attn_prob)):
#             torch.nn.init.normal_(expected_attn_prob[i], mean=0.0, std=1.0)
#
#         assert output_tensor.shape == expected_output_tensor.shape
#         assert torch.allclose(output_tensor, expected_output_tensor, atol=1e-6)
#         assert all(torch.allclose(attn_prob[i], expected_attn_prob[i], atol=1e-6) for i in range(len(attn_prob)))
#
#
# class TestCrossAttention:
#     @classmethod
#     def setup_class(cls):
#         cls.n_heads = 2
#         cls.d_model = 4
#         cls.d_memory = 6
#         cls.p_dropout = 0.1
#
#     def test_forward(self):
#         layer = CrossAttention(self.n_heads, self.d_model, self.d_memory, self.p_dropout)
#         input_tensor = torch.randn(1, 10, self.d_model)
#         memory_tensor = torch.randn(1, 15, self.d_memory)
#         mask_tensor = torch.ones(1, 10).bool()
#         memory_mask = torch.ones(1, 15).bool()
#         output_tensor, attn_prob = layer(input_tensor, mask_tensor, memory_tensor, memory_mask)
#
#         # Mock expected output tensor
#         expected_output_tensor = torch.randn(1, 10, self.d_model)
#         expected_attn_prob = [torch.randn(1, self.n_heads, 10, 15), torch.randn(1, self.n_heads, 10, 15)]
#         torch.manual_seed(0)  # Ensure reproducibility
#         torch.nn.init.normal_(expected_output_tensor, mean=0.0, std=1.0)
#         for i in range(len(expected_attn_prob)):
#             torch.nn.init.normal_(expected_attn_prob[i], mean=0.0, std=1.0)
#
#         assert output_tensor.shape == expected_output_tensor.shape
#         assert torch.allclose(output_tensor, expected_output_tensor, atol=1e-6)
#         assert all(torch.allclose(attn_prob[i], expected_attn_prob[i], atol=1e-6) for i in range(len(attn_prob)))
#
#
# class TestTransformerLayer:
#     @classmethod
#     def setup_class(cls):
#         cls.d_model = 4
#         cls.d_ffn = 16
#         cls.sa_n_heads = 2
#         cls.kernel_size = 3
#         cls.p_dropout = 0.1
#         cls.has_xattn = False
#         cls.is_causal = True
#         cls.max_length_causal_mask = 10
#
#     def test_forward(self):
#         layer = TransformerLayer(
#             self.d_model,
#             self.d_ffn,
#             self.sa_n_heads,
#             self.kernel_size,
#             self.p_dropout,
#             self.has_xattn,
#             is_causal=self.is_causal,
#             max_length_causal_mask=self.max_length_causal_mask,
#         )
#         input_tensor = torch.randn(1, 10, self.d_model)
#         mask_tensor = torch.ones(1, 10).bool()
#         output_dict = layer(input_tensor, mask_tensor)
#
#         # Mock expected output tensor
#         expected_output_tensor = torch.randn(1, 10, self.d_model)
#         torch.manual_seed(0)  # Ensure reproducibility
#         torch.nn.init.normal_(expected_output_tensor, mean=0.0, std=1.0)
#
#         assert output_dict['output'].shape == expected_output_tensor.shape
#         assert torch.allclose(output_dict['output'], expected_output_tensor, atol=1e-6)


# @pytest.mark.run_only_on('CPU')
@pytest.mark.unit
class TestTransformer:
    @classmethod
    def setup_class(cls):
        cls.n_layers = 1
        cls.d_model = 4
        cls.d_ffn = 16
        cls.sa_n_heads = 2
        cls.kernel_size = 3
        cls.p_dropout = 0.0
        cls.p_dropout_out = 0.0
        # cls.has_xattn = False
        cls.is_causal = True
        cls.max_length_causal_mask = 10

        # fmt:off
        cls.input_tensor = torch.Tensor(
            [[[ 0.7049,  0.0305, -0.8542,  0.5388],
              [-0.5265, -1.3320,  1.5451,  0.4086],
              [-2.0546,  0.5259,  0.5995, -0.4078],
              [ 0.4530, -0.3918,  2.1403, -0.2062],
              [-0.0984,  0.4855,  0.7076,  0.0431],
              [-0.4394, -0.6761,  1.7389, -0.9423],
              [ 0.9764, -1.0889, -0.1634,  2.2799],
              [ 0.2277,  0.0367,  0.3680,  0.9759],
              [ 0.8760,  1.4248, -0.2724,  0.9353],
              [-1.4920, -0.5683, -0.9277,  2.1160]]]
        )
        # fmt:on

    def test_forward_causal_self_attn_and_no_xattn(self):
        set_seed(0)
        model = Transformer(
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            sa_n_heads=self.sa_n_heads,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout,
            p_dropout_out=self.p_dropout_out,
            has_xattn=False,
            is_causal=self.is_causal,
            max_length_causal_mask=self.max_length_causal_mask,
        )

        # # fmt:off
        # input_tensor = torch.Tensor(
        #     [[[ 0.7049,  0.0305, -0.8542,  0.5388],
        #       [-0.5265, -1.3320,  1.5451,  0.4086],
        #       [-2.0546,  0.5259,  0.5995, -0.4078],
        #       [ 0.4530, -0.3918,  2.1403, -0.2062],
        #       [-0.0984,  0.4855,  0.7076,  0.0431],
        #       [-0.4394, -0.6761,  1.7389, -0.9423],
        #       [ 0.9764, -1.0889, -0.1634,  2.2799],
        #       [ 0.2277,  0.0367,  0.3680,  0.9759],
        #       [ 0.8760,  1.4248, -0.2724,  0.9353],
        #       [-1.4920, -0.5683, -0.9277,  2.1160]]]
        # )
        # # fmt:on

        mask_tensor = torch.zeros(1, self.max_length_causal_mask).bool()
        with torch.no_grad():
            output_dict = model(x=self.input_tensor, x_mask=mask_tensor)

        # Mock expected output tensor
        # fmt:off
        expected_output_tensor = {
            'output': torch.Tensor(
                [[[ 0.6969,  0.0227, -0.8543,  0.5439],
                  [-0.5119, -1.3091,  1.5798,  0.3773],
                  [-2.0538,  0.4955,  0.5473, -0.3819],
                  [ 0.5036, -0.3710,  2.1465, -0.1668],
                  [-0.0887,  0.4619,  0.7054,  0.0682],
                  [-0.4506, -0.6643,  1.7608, -0.9786],
                  [ 0.9630, -1.1201, -0.1460,  2.2891],
                  [ 0.2344,  0.0400,  0.4199,  0.9065],
                  [ 0.8862,  1.4106, -0.3032,  0.9565],
                  [-1.4381, -0.5293, -0.8979,  2.0783]]]
            ),
            'attn_probabilities': [
                {
                    'self_attn_probabilities': [
                        torch.Tensor(
                            [[[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.4998, 0.5002, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.3337, 0.3333, 0.3330, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.2498, 0.2500, 0.2501, 0.2501, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.2002, 0.2001, 0.2000, 0.1999, 0.1999, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.1666, 0.1666, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.1427, 0.1428, 0.1429, 0.1430, 0.1430, 0.1430, 0.1427, 0.0000, 0.0000, 0.0000],
                               [0.1250, 0.1250, 0.1250, 0.1251, 0.1250, 0.1251, 0.1249, 0.1249, 0.0000, 0.0000],
                               [0.1112, 0.1111, 0.1111, 0.1110, 0.1110, 0.1110, 0.1112, 0.1112, 0.1111, 0.0000],
                               [0.1001, 0.1000, 0.1000, 0.1000, 0.0999, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000]],
                              [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.5005, 0.4995, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.3332, 0.3331, 0.3336, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.2507, 0.2494, 0.2510, 0.2489, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.2002, 0.1995, 0.2006, 0.1994, 0.2003, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.1671, 0.1663, 0.1674, 0.1660, 0.1670, 0.1662, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.1427, 0.1432, 0.1423, 0.1432, 0.1426, 0.1431, 0.1430, 0.0000, 0.0000, 0.0000],
                               [0.1248, 0.1253, 0.1245, 0.1254, 0.1248, 0.1253, 0.1251, 0.1249, 0.0000, 0.0000],
                               [0.1109, 0.1114, 0.1108, 0.1116, 0.1111, 0.1115, 0.1111, 0.1110, 0.1106, 0.0000],
                               [0.0999, 0.1004, 0.0996, 0.1006, 0.0999, 0.1005, 0.1001, 0.0999, 0.0994, 0.0996]]]]
                        ),
                        torch.Tensor(
                            [[[[-3.4823e-04, float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                               [-7.0210e-04, -5.6984e-05, float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                               [1.2551e-03, 1.1431e-04, -5.9334e-04, float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                               [-8.1514e-04, 2.6650e-05, 5.2952e-04, 5.6903e-04, float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                               [8.0150e-04, 1.4366e-04, -2.7793e-04, -7.0636e-04, -8.2140e-04, float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                               [-4.6137e-04, 6.2648e-05, 3.6768e-04, 2.8095e-04, 4.9188e-04, 3.6888e-04, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                               [-8.4568e-04, -1.8879e-04, 2.4008e-04, 7.7748e-04, 8.6180e-04, 8.9809e-04, -1.3661e-03, float("-inf"), float("-inf"), float("-inf")],
                               [-2.7906e-04, -1.5744e-04, -5.6727e-05, 3.3880e-04, 2.7191e-04, 3.6589e-04, -5.9697e-04, -7.4063e-04, float("-inf"), float("-inf")],
                               [8.2735e-04, 7.1352e-06, -4.8861e-04, -6.0711e-04, -8.6640e-04, -7.4883e-04, 1.0636e-03, 9.7322e-04, 3.8468e-04, float("-inf")],
                               [5.1061e-04, -1.1017e-04, -4.6527e-04, -2.7564e-04, -5.4973e-04, -3.7841e-04, 4.8037e-04, 2.9588e-04, 2.1742e-04, 2.2418e-04]],
                              [[-7.5861e-04, float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                               [8.0373e-04, -1.2745e-03, float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                               [-4.5038e-04, -6.7648e-04, 7.5978e-04, float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                               [1.5376e-03, -3.6984e-03, 3.0423e-03, -5.4870e-03, float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                               [3.7014e-04, -2.7010e-03, 2.4310e-03, -3.2604e-03, 9.3840e-04, float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                               [1.3868e-03, -3.8372e-03, 3.2144e-03, -5.4860e-03, 5.0273e-04, -4.4343e-03, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                               [-4.0899e-04, 3.0823e-03, -2.7779e-03, 3.7073e-03, -1.0838e-03, 2.7842e-03, 1.7633e-03, float("-inf"), float("-inf"), float("-inf")],
                               [-7.3482e-04, 3.2334e-03, -2.8291e-03, 4.1924e-03, -8.4200e-04, 3.2580e-03, 1.4279e-03, 8.5064e-05, float("-inf"), float("-inf")],
                               [-1.3268e-03, 2.9038e-03, -2.3554e-03, 4.4267e-03, -1.1294e-04, 3.6616e-03, 3.6230e-04, -9.0338e-04, -4.0710e-03, float("-inf")],
                               [ -1.4350e-03, 4.1491e-03, -3.4936e-03, 5.8679e-03, -6.0590e-04, 4.7234e-03, 1.1535e-03, -6.1370e-04, -5.5221e-03, -4.1614e-03]]]]
                        ),
                    ],
                    'cross_attn_probabilities': None,
                }
            ],
        }
        # fmt:on

        assert output_dict["output"].shape == expected_output_tensor["output"].shape
        assert torch.allclose(output_dict["output"], expected_output_tensor["output"], atol=1e-4)
        for i in range(2):
            assert torch.allclose(
                output_dict["attn_probabilities"][0]["self_attn_probabilities"][i],
                expected_output_tensor["attn_probabilities"][0]["self_attn_probabilities"][i],
                atol=1e-4,
            )
        assert output_dict["attn_probabilities"][0]["cross_attn_probabilities"] is None

    def test_forward_causal_self_attn_and_has_xattn(self):
        set_seed(0)
        model = Transformer(
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            sa_n_heads=self.sa_n_heads,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout,
            p_dropout_out=self.p_dropout_out,
            has_xattn=True,
            xa_d_memory=4,
            xa_n_heads=2,
            is_causal=self.is_causal,
            max_length_causal_mask=self.max_length_causal_mask,
        )

        # fmt:off
        # input_tensor = torch.Tensor(
        #     [[[ 0.7049,  0.0305, -0.8542,  0.5388],
        #       [-0.5265, -1.3320,  1.5451,  0.4086],
        #       [-2.0546,  0.5259,  0.5995, -0.4078],
        #       [ 0.4530, -0.3918,  2.1403, -0.2062],
        #       [-0.0984,  0.4855,  0.7076,  0.0431],
        #       [-0.4394, -0.6761,  1.7389, -0.9423],
        #       [ 0.9764, -1.0889, -0.1634,  2.2799],
        #       [ 0.2277,  0.0367,  0.3680,  0.9759],
        #       [ 0.8760,  1.4248, -0.2724,  0.9353],
        #       [-1.4920, -0.5683, -0.9277,  2.1160]]]
        # )
        cond = [
            # shape (1, 6, 2)
            torch.Tensor(
                [[[1.0101, -0.4045],
                  [2.1502, 0.3137],
                  [1.0659, -0.6845],
                  [-0.6656, -1.9148],
                  [0.3510, 0.7737],
                  [-0.2495, -1.0554]]]
            ),
            # shape (1, 5, 3)
            torch.Tensor(
                [[[0.2122, -0.5399, -0.0741],
                  [0.2523, -0.4742, 0.9657],
                  [-1.3595, 2.1364, 1.2012],
                  [1.1759, -1.5697, -0.3139],
                  [0.8711, -0.6026, 0.4475]]]
            )
        ]
        # fmt:on

        cond_mask = [torch.zeros(1, 6).bool(), torch.zeros(1, 5).bool()]

        mask_tensor = torch.zeros(1, 10).bool()
        with torch.no_grad():
            output_dict = model(x=self.input_tensor, x_mask=mask_tensor, cond=cond, cond_mask=cond_mask)

        print(output_dict)
