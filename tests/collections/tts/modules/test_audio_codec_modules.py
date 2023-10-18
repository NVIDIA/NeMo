# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pytest
import torch

from nemo.collections.tts.modules.audio_codec_modules import (
    CodecActivation,
    Conv1dNorm,
    ConvTranspose1dNorm,
    FiniteScalarQuantizer,
    GroupFiniteScalarQuantizer,
    get_down_sample_padding,
)
from nemo.collections.tts.modules.encodec_modules import GroupResidualVectorQuantizer, ResidualVectorQuantizer


class TestAudioCodecModules:
    def setup_class(self):
        self.in_channels = 8
        self.out_channels = 16
        self.batch_size = 2
        self.len1 = 4
        self.len2 = 8
        self.max_len = 10
        self.kernel_size = 3

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_conv1d(self):
        inputs = torch.rand([self.batch_size, self.in_channels, self.max_len])
        lengths = torch.tensor([self.len1, self.len2], dtype=torch.int32)

        conv = Conv1dNorm(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size)
        out = conv(inputs=inputs, input_len=lengths)

        assert out.shape == (self.batch_size, self.out_channels, self.max_len)
        assert torch.all(out[0, :, : self.len1] != 0.0)
        assert torch.all(out[0, :, self.len1 :] == 0.0)
        assert torch.all(out[1, :, : self.len2] != 0.0)
        assert torch.all(out[1, :, self.len2 :] == 0.0)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_conv1d_downsample(self):
        stride = 2
        out_len = self.max_len // stride
        out_len_1 = self.len1 // stride
        out_len_2 = self.len2 // stride
        inputs = torch.rand([self.batch_size, self.in_channels, self.max_len])
        lengths = torch.tensor([out_len_1, out_len_2], dtype=torch.int32)

        padding = get_down_sample_padding(kernel_size=self.kernel_size, stride=stride)
        conv = Conv1dNorm(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
        )
        out = conv(inputs=inputs, input_len=lengths)

        assert out.shape == (self.batch_size, self.out_channels, out_len)
        assert torch.all(out[0, :, :out_len_1] != 0.0)
        assert torch.all(out[0, :, out_len_1:] == 0.0)
        assert torch.all(out[1, :, :out_len_2] != 0.0)
        assert torch.all(out[1, :, out_len_2:] == 0.0)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_conv1d_transpose_upsample(self):
        stride = 2
        out_len = self.max_len * stride
        out_len_1 = self.len1 * stride
        out_len_2 = self.len2 * stride
        inputs = torch.rand([self.batch_size, self.in_channels, self.max_len])
        lengths = torch.tensor([out_len_1, out_len_2], dtype=torch.int32)

        conv = ConvTranspose1dNorm(
            in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=stride
        )
        out = conv(inputs=inputs, input_len=lengths)

        assert out.shape == (self.batch_size, self.out_channels, out_len)
        assert torch.all(out[0, :, :out_len_1] != 0.0)
        assert torch.all(out[0, :, out_len_1:] == 0.0)
        assert torch.all(out[1, :, :out_len_2] != 0.0)
        assert torch.all(out[1, :, out_len_2:] == 0.0)


class TestResidualVectorQuantizer:
    def setup_class(self):
        """Setup common members
        """
        self.batch_size = 2
        self.max_len = 20
        self.codebook_size = 256
        self.codebook_dim = 64
        self.num_examples = 10

    @pytest.mark.unit
    @pytest.mark.parametrize('num_codebooks', [1, 4])
    def test_rvq_eval(self, num_codebooks: int):
        """Simple test to confirm that the RVQ module can be instantiated and run,
        and that forward produces the same result as encode-decode.
        """
        # instantiate and set in eval mode
        rvq = ResidualVectorQuantizer(num_codebooks=num_codebooks, codebook_dim=self.codebook_dim)
        rvq.eval()

        for i in range(self.num_examples):
            inputs = torch.randn([self.batch_size, self.codebook_dim, self.max_len])
            input_len = torch.tensor([self.max_len] * self.batch_size, dtype=torch.int32)

            # apply forward
            dequantized_fw, indices_fw, commit_loss = rvq(inputs=inputs, input_len=input_len)

            # make sure the commit loss is zero
            assert commit_loss == 0.0, f'example {i}: commit_loss is {commit_loss}, expected 0.0'

            # encode-decode
            indices_enc = rvq.encode(inputs=inputs, input_len=input_len)
            dequantized_dec = rvq.decode(indices=indices_enc, input_len=input_len)

            # make sure the results are the same
            torch.testing.assert_close(indices_enc, indices_fw, msg=f'example {i}: indices mismatch')
            torch.testing.assert_close(dequantized_dec, dequantized_fw, msg=f'example {i}: dequantized mismatch')

    @pytest.mark.unit
    @pytest.mark.parametrize('num_groups', [1, 2, 4])
    @pytest.mark.parametrize('num_codebooks', [1, 4])
    def test_group_rvq_eval(self, num_groups: int, num_codebooks: int):
        """Simple test to confirm that the group RVQ module can be instantiated and run,
        and that forward produces the same result as encode-decode.
        """
        if num_groups > num_codebooks:
            # Expected to fail if num_groups is lager than the total number of codebooks
            with pytest.raises(ValueError):
                _ = GroupResidualVectorQuantizer(
                    num_codebooks=num_codebooks, num_groups=num_groups, codebook_dim=self.codebook_dim
                )
        else:
            # Test inference with group RVQ
            # instantiate and set in eval mode
            grvq = GroupResidualVectorQuantizer(
                num_codebooks=num_codebooks, num_groups=num_groups, codebook_dim=self.codebook_dim
            )
            grvq.eval()

            for i in range(self.num_examples):
                inputs = torch.randn([self.batch_size, self.codebook_dim, self.max_len])
                input_len = torch.tensor([self.max_len] * self.batch_size, dtype=torch.int32)

                # apply forward
                dequantized_fw, indices_fw, commit_loss = grvq(inputs=inputs, input_len=input_len)

                # make sure the commit loss is zero
                assert commit_loss == 0.0, f'example {i}: commit_loss is {commit_loss}, expected 0.0'

                # encode-decode
                indices_enc = grvq.encode(inputs=inputs, input_len=input_len)
                dequantized_dec = grvq.decode(indices=indices_enc, input_len=input_len)

                # make sure the results are the same
                torch.testing.assert_close(indices_enc, indices_fw, msg=f'example {i}: indices mismatch')
                torch.testing.assert_close(dequantized_dec, dequantized_fw, msg=f'example {i}: dequantized mismatch')

                # apply individual RVQs and make sure the results are the same
                inputs_grouped = inputs.chunk(num_groups, dim=1)
                dequantized_fw_grouped = dequantized_fw.chunk(num_groups, dim=1)
                indices_fw_grouped = indices_fw.chunk(num_groups, dim=0)

                for g in range(num_groups):
                    dequantized, indices, _ = grvq.rvqs[g](inputs=inputs_grouped[g], input_len=input_len)
                    torch.testing.assert_close(
                        dequantized, dequantized_fw_grouped[g], msg=f'example {i}: dequantized mismatch for group {g}'
                    )
                    torch.testing.assert_close(
                        indices, indices_fw_grouped[g], msg=f'example {i}: indices mismatch for group {g}'
                    )


class TestCodecActivation:
    def setup_class(self):
        self.batch_size = 2
        self.in_channels = 4
        self.max_len = 4

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_snake(self):
        """
        Test for snake activation function execution.
        """
        inputs = torch.rand([self.batch_size, self.in_channels, self.max_len])
        snake = CodecActivation('snake', channels=self.in_channels)
        out = snake(x=inputs)
        assert out.shape == (self.batch_size, self.in_channels, self.max_len)


class TestFiniteScalarQuantizer:
    def setup_class(self):
        """Setup common members
        """
        self.batch_size = 2
        self.max_len = 20
        self.num_examples = 10

    @pytest.mark.unit
    @pytest.mark.parametrize('num_levels', [[2, 3], [8, 5, 5]])
    def test_fsq_eval(self, num_levels: list):
        """Simple test to confirm that the FSQ module can be instantiated and run,
        and that forward produces the same result as encode-decode.
        """
        fsq = FiniteScalarQuantizer(num_levels=num_levels)

        for i in range(self.num_examples):
            inputs = torch.randn([self.batch_size, fsq.codebook_dim, self.max_len])
            input_len = torch.tensor([self.max_len] * self.batch_size, dtype=torch.int32)

            # apply forward
            dequantized_fw, indices_fw = fsq(inputs=inputs, input_len=input_len)

            assert dequantized_fw.max() <= 1.0, f'example {i}: dequantized_fw.max() is {dequantized_fw.max()}'
            assert dequantized_fw.min() >= -1.0, f'example {i}: dequantized_fw.min() is {dequantized_fw.min()}'

            # encode-decode
            indices_enc = fsq.encode(inputs=inputs, input_len=input_len)
            dequantized_dec = fsq.decode(indices=indices_enc, input_len=input_len)

            # make sure the results are the same
            torch.testing.assert_close(indices_enc, indices_fw, msg=f'example {i}: indices mismatch')
            torch.testing.assert_close(dequantized_dec, dequantized_fw, msg=f'example {i}: dequantized mismatch')

    @pytest.mark.unit
    def test_fsq_output(self):
        """Simple test to make sure the output of FSQ is correct
        for a single setup.
        """
        num_levels = [2, 3]
        fsq = FiniteScalarQuantizer(num_levels=num_levels)

        # # To generate inputs & outputs for testing
        # max_len = 8
        # inputs = torch.randn([self.num_examples, fsq.codebook_dim, max_len])
        # input_len = torch.tensor([max_len] * self.num_examples, dtype=torch.int32)
        # dequantized, indices = fsq(inputs=inputs, input_len=input_len)
        # print(inputs)
        # print(input_len)
        # print(dequantized)
        # print(indices)

        # inputs
        inputs = torch.tensor(
            [
                [
                    [0.6572, 1.3574, 2.1646, 0.9457, -0.3489, 0.6732, -0.7148, -0.5143],
                    [0.5123, 0.8552, 1.7814, 1.9938, -1.1909, -0.9991, -3.7932, -0.4438],
                ],
                [
                    [0.2357, 0.8324, 0.8932, -0.0596, 0.6130, -0.0299, 0.3824, 1.6278],
                    [-0.3781, 0.1864, -0.2190, 1.2199, -1.1398, -0.8443, -0.7865, 0.1470],
                ],
                [
                    [0.9786, 1.2170, 0.2229, -0.6481, -0.0348, 0.0552, 0.3956, -1.0916],
                    [1.2982, 0.5188, 0.3546, -1.5305, -1.0674, 1.1292, 0.7662, -0.1397],
                ],
                [
                    [-1.8530, -0.3099, 1.7705, 0.2201, 0.3348, -0.2126, 0.4756, -0.9759],
                    [-0.6812, 0.8368, -0.5181, -0.6713, -0.0681, -1.5496, 0.9230, 0.3448],
                ],
                [
                    [-1.1940, -1.3429, 0.5648, 1.5043, -1.0501, -1.0594, -1.0261, -0.2600],
                    [-1.3521, -0.4740, -1.6166, -0.8975, 0.6101, 0.2225, 1.0959, -0.0723],
                ],
                [
                    [-1.4756, 0.0630, -1.9273, 0.6048, 0.0432, -0.4550, -0.9183, 1.4493],
                    [-0.8326, 0.4620, 1.8287, 0.2323, -1.5944, -0.8721, 0.0126, -2.1843],
                ],
                [
                    [0.0913, -0.4664, 0.2600, 0.7711, 0.7383, 0.8726, 0.1065, 0.5777],
                    [-0.9217, 1.9214, -0.4060, 1.5786, 0.5549, 1.6364, 0.2880, -0.7962],
                ],
                [
                    [-0.5967, -1.0998, -0.2475, 1.2475, -2.1949, 0.0607, 0.5634, 1.0397],
                    [0.2047, 0.3775, -0.1769, 0.7248, -1.6236, 0.5641, 0.9344, 0.2959],
                ],
                [
                    [0.2875, 0.6632, -0.6974, -0.0710, 0.1296, 0.0872, -1.0767, 1.3350],
                    [0.7032, 0.6264, -0.3976, -0.0257, -0.5352, -1.1119, -1.0472, 0.1626],
                ],
                [
                    [0.2453, -0.9992, 0.3687, 1.5681, -0.3206, 0.2046, -0.4617, 0.7606],
                    [-1.8594, 0.2092, 0.1827, -1.1598, 0.0664, 0.9267, 1.5458, 1.5875],
                ],
            ]
        )

        input_len = torch.tensor([8, 8, 8, 8, 8, 8, 8, 8, 8, 8], dtype=torch.int32)

        # expected output
        dequantized_expected = torch.tensor(
            [
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0, -1.0, -1.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, -1.0, -1.0, 1.0, 1.0, 0.0]],
                [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 1.0, 0.0, -1.0, 0.0, -1.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, -1.0, -1.0, 1.0, 0.0, 1.0, 0.0]],
                [[0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0, -1.0]],
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, -1.0]],
                [[0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 1.0]],
            ]
        )

        indices_expected = torch.tensor(
            [
                [
                    [3, 5, 5, 5, 1, 1, 1, 3],
                    [3, 3, 3, 5, 1, 1, 1, 3],
                    [5, 3, 3, 1, 1, 5, 5, 3],
                    [0, 5, 3, 1, 3, 1, 5, 3],
                    [1, 3, 1, 1, 5, 3, 5, 3],
                    [1, 3, 4, 3, 1, 1, 3, 1],
                    [1, 5, 3, 5, 5, 5, 3, 1],
                    [3, 3, 3, 5, 0, 5, 5, 3],
                    [5, 5, 3, 3, 3, 1, 1, 3],
                    [1, 3, 3, 1, 3, 5, 5, 5],
                ]
            ],
            dtype=torch.int32,
        )

        # test
        dequantized, indices = fsq(inputs=inputs, input_len=input_len)
        torch.testing.assert_close(dequantized, dequantized_expected, msg=f'dequantized mismatch')
        torch.testing.assert_close(indices, indices_expected, msg=f'indices mismatch')

    @pytest.mark.unit
    @pytest.mark.parametrize('num_groups', [1, 2, 4])
    @pytest.mark.parametrize('num_levels_per_group', [[2, 3], [8, 5, 5]])
    def test_group_fsq_eval(self, num_groups: int, num_levels_per_group: int):
        """Simple test to confirm that the group FSQ module can be instantiated and run,
        and that forward produces the same result as encode-decode.
        """
        # Test inference with group FSQ
        # instantiate
        gfsq = GroupFiniteScalarQuantizer(num_groups=num_groups, num_levels_per_group=num_levels_per_group)

        for i in range(self.num_examples):
            inputs = torch.randn([self.batch_size, gfsq.codebook_dim, self.max_len])
            input_len = torch.tensor([self.max_len] * self.batch_size, dtype=torch.int32)

            # apply forward
            dequantized_fw, indices_fw = gfsq(inputs=inputs, input_len=input_len)

            # encode-decode
            indices_enc = gfsq.encode(inputs=inputs, input_len=input_len)
            dequantized_dec = gfsq.decode(indices=indices_enc, input_len=input_len)

            # make sure the results are the same
            torch.testing.assert_close(indices_enc, indices_fw, msg=f'example {i}: indices mismatch')
            torch.testing.assert_close(dequantized_dec, dequantized_fw, msg=f'example {i}: dequantized mismatch')

            # apply individual FSQs and make sure the results are the same
            inputs_grouped = inputs.chunk(num_groups, dim=1)
            dequantized_fw_grouped = dequantized_fw.chunk(num_groups, dim=1)
            indices_fw_grouped = indices_fw.chunk(num_groups, dim=0)

            for g in range(num_groups):
                dequantized, indices = gfsq.fsqs[g](inputs=inputs_grouped[g], input_len=input_len)
                torch.testing.assert_close(
                    dequantized, dequantized_fw_grouped[g], msg=f'example {i}: dequantized mismatch for group {g}'
                )
                torch.testing.assert_close(
                    indices, indices_fw_grouped[g], msg=f'example {i}: indices mismatch for group {g}'
                )
