# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
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

from unittest.mock import MagicMock

import pytest
import torch
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils import B2BCausalConv1dModule


class MockProjConv(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.short_conv_weight = torch.randn(1, 1, kernel_size)
        self.group_dim = 1


class MockMixer(torch.nn.Module):
    def __init__(self, kernel_size, use_conv_bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.hyena_medium_conv_len = 10
        self.use_conv_bias = use_conv_bias
        self.group_dim = 1
        # Create a mock short_conv module
        self.short_conv = torch.nn.Module()
        self.short_conv.short_conv_weight = torch.randn(1, 1, kernel_size)
        self.short_conv.kernel_size = kernel_size
        # conv_bias attribute for bias handling
        self.conv_bias = torch.randn(1) if use_conv_bias else None
        # Create a mock filter function
        self.filter = MagicMock()
        self.filter.return_value = (torch.randn(1, 1, kernel_size), torch.randn(1, 1, kernel_size))


def mock_b2b_causal_conv1d(x, weight_proj, weight_mixer, skip_bias):
    """Mock implementation of b2b_causal_conv1d that returns only the tensor for test slicing."""

    return x


@pytest.mark.parametrize("operator_type", ["hyena_short_conv", "hyena_medium_conv"])
def test_b2b_causal_conv1d_module_initialization(operator_type):
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5)

    b2b_module = B2BCausalConv1dModule(proj_conv, mixer, operator_type=operator_type)

    assert b2b_module.operator_type == operator_type
    assert b2b_module._proj_conv_module == proj_conv
    assert b2b_module._mixer_module == mixer


@pytest.mark.parametrize("operator_type", ["hyena_short_conv", "hyena_medium_conv"])
def test_b2b_causal_conv1d_module_weight_extraction(operator_type):
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5)
    b2b_module = B2BCausalConv1dModule(proj_conv, mixer, operator_type=operator_type, b2b_causal_conv1d=mock_b2b_causal_conv1d)
    x = torch.randn(2, 96, 10)  # [B, D, L]
    result = b2b_module(x)

    assert result.shape == x.shape


@pytest.mark.parametrize("operator_type", ["hyena_short_conv", "hyena_medium_conv"])
@pytest.mark.parametrize("use_conv_bias", [True, False])
def test_b2b_causal_conv1d_module_bias_handling(use_conv_bias, operator_type):
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5, use_conv_bias=use_conv_bias)
    b2b_module = B2BCausalConv1dModule(proj_conv, mixer, operator_type=operator_type, b2b_causal_conv1d=mock_b2b_causal_conv1d)
    x = torch.randn(2, 96, 10)  # [B, D, L]
    result = b2b_module(x)

    assert result.shape == x.shape


def test_b2b_causal_conv1d_module_invalid_operator():
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5)

    with pytest.raises(ValueError, match="Operator type invalid_type not supported"):
        B2BCausalConv1dModule(proj_conv, mixer, operator_type="invalid_type")


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [8, 16, 32])
def test_b2b_causal_conv1d_module_different_shapes(batch_size, seq_len):
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5)
    b2b_module = B2BCausalConv1dModule(proj_conv, mixer, operator_type="hyena_short_conv", b2b_causal_conv1d=mock_b2b_causal_conv1d)

    # Test with different hidden dimensions
    for hidden_dim in [32, 64, 128]:
        x = torch.randn(batch_size, hidden_dim, seq_len)
        result = b2b_module(x)
        assert result.shape == x.shape, f"Shape mismatch for batch_size={batch_size}, hidden_dim={hidden_dim}, seq_len={seq_len}"


@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_b2b_causal_conv1d_module_different_kernel_sizes(kernel_size):
    proj_conv = MockProjConv(kernel_size=kernel_size)
    mixer = MockMixer(kernel_size=kernel_size)
    b2b_module = B2BCausalConv1dModule(proj_conv, mixer, operator_type="hyena_short_conv", b2b_causal_conv1d=mock_b2b_causal_conv1d)
    x = torch.randn(2, 96, 32)
    result = b2b_module(x)

    assert result.shape == x.shape, f"Shape mismatch for kernel_size={kernel_size}"


def test_b2b_causal_conv1d_module_invalid_input():
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5)
    b2b_module = B2BCausalConv1dModule(proj_conv, mixer, operator_type="hyena_short_conv", b2b_causal_conv1d=mock_b2b_causal_conv1d)

    # Test with invalid input dimensions
    with pytest.raises(ValueError, match="Input tensor must be 3D"):
        b2b_module(torch.randn(2, 96))  # Missing sequence dimension


def test_b2b_causal_conv1d_module_dtype_handling():
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5)
    b2b_module = B2BCausalConv1dModule(proj_conv, mixer, operator_type="hyena_short_conv", b2b_causal_conv1d=mock_b2b_causal_conv1d)

    # Test with different dtypes
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    for dtype in dtypes:
        x = torch.randn(2, 96, 32, dtype=dtype)
        result = b2b_module(x)

        assert result.dtype == dtype, f"Dtype mismatch for {dtype}"


def test_b2b_causal_conv1d_module_device_handling():
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5)
    b2b_module = B2BCausalConv1dModule(proj_conv, mixer, operator_type="hyena_short_conv", b2b_causal_conv1d=mock_b2b_causal_conv1d)

    # Test on CPU
    x_cpu = torch.randn(2, 96, 32)
    result_cpu = b2b_module(x_cpu)
    assert result_cpu.device == x_cpu.device, "Device mismatch on CPU"

    # Test on CUDA if available
    if torch.cuda.is_available():
        x_cuda = x_cpu.cuda()
        result_cuda = b2b_module(x_cuda)
        assert result_cuda.device == x_cuda.device, "Device mismatch on CUDA"


def test_b2b_causal_conv1d_module_gradient_flow():
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5)
    b2b_module = B2BCausalConv1dModule(proj_conv, mixer, operator_type="hyena_short_conv", b2b_causal_conv1d=mock_b2b_causal_conv1d)

    x = torch.randn(2, 96, 32, requires_grad=True)
    result = b2b_module(x)

    # Test backward pass
    loss = result.mean()
    loss.backward()

    # Check if gradients are computed
    assert x.grad is not None, "Gradients not computed for input"
    assert x.grad.shape == x.shape, "Gradient shape mismatch"


def test_b2b_causal_conv1d_module_requires_grad():
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5)
    b2b_module = B2BCausalConv1dModule(proj_conv, mixer, operator_type="hyena_short_conv", b2b_causal_conv1d=mock_b2b_causal_conv1d)

    # Test with requires_grad=True
    x = torch.randn(2, 96, 32, requires_grad=True)
    result = b2b_module(x)

    assert result.requires_grad, "Output should require gradients when input requires gradients"

    # Test with requires_grad=False
    x = torch.randn(2, 96, 32, requires_grad=False)
    result = b2b_module(x)

    assert not result.requires_grad, "Output should not require gradients when input doesn't require gradients"
