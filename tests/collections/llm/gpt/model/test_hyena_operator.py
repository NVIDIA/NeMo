# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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

import os
from contextlib import contextmanager
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils import (
    ParallelCausalDepthwiseConv1d,
    ParallelHyenaOperator,
    ParallelShortHyenaOperator,
)


@contextmanager
def simple_parallel_state():
    """Setup for parallel state testing - mimics the working test."""
    try:
        # Clean up any existing state
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()

        # Set up environment variables
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        # Initialize process group
        if not dist.is_initialized():
            timeout_timedelta = timedelta(seconds=1800)
            dist.init_process_group(backend="nccl", timeout=timeout_timedelta)

        # Initialize parallel state
        parallel_state.initialize_model_parallel()

        # Initialize the model parallel RNG
        model_parallel_cuda_manual_seed(42)

        yield

    finally:
        # Clean up
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture
def hyena_config() -> HyenaConfig:
    return HyenaConfig()


@pytest.fixture
def transformer_config() -> TransformerConfig:
    return TransformerConfig(num_layers=2, hidden_size=864, num_attention_heads=1)


class TestParallelHyenaOperator:
    @pytest.fixture
    def operator(self, transformer_config: TransformerConfig, hyena_config: HyenaConfig) -> ParallelHyenaOperator:
        with simple_parallel_state():
            yield ParallelHyenaOperator(
                hidden_size=transformer_config.hidden_size,
                transformer_config=transformer_config,
                hyena_config=hyena_config,
                max_sequence_length=1024,
                operator_type="hyena_medium_conv",
                init_method="small_init",
            )

    def test_initialization(self, operator: ParallelHyenaOperator):
        assert operator.hidden_size == 864
        assert operator.operator_type == "hyena_medium_conv"
        assert isinstance(operator.conv_bias, torch.nn.Parameter)
        num_weights = sum([p.numel() for p in operator.parameters()])
        assert num_weights == 111456

    def test_gpu_forward(self, operator: ParallelHyenaOperator):
        device = torch.device("cuda")
        operator = operator.to(device)
        batch_size = 2
        seq_len = operator.L  # operator.L maps to max_sequence_length
        g = operator.num_groups
        dg = operator.group_dim

        x1 = torch.ones((batch_size, (g * dg), seq_len), device=device)
        x2 = torch.ones((batch_size, (g * dg), seq_len), device=device)
        v = torch.ones((batch_size, (g * dg), seq_len), device=device)

        output = operator(x1, x2, v)
        assert output.shape[0] == batch_size
        assert output.shape[1] == operator.hidden_size
        assert output.shape[2] == seq_len


class TestParallelShortHyenaOperator:
    @pytest.fixture
    def operator(self, transformer_config: TransformerConfig, hyena_config: HyenaConfig) -> ParallelShortHyenaOperator:
        with simple_parallel_state():
            yield ParallelShortHyenaOperator(
                hidden_size=transformer_config.hidden_size,
                transformer_config=transformer_config,
                hyena_config=hyena_config,
                init_method="small_init",
                short_conv_class=ParallelCausalDepthwiseConv1d,
                use_fast_causal_conv=False,
                local_init=False,
                use_conv_bias=False,
            )

    def test_initialization(self, operator: ParallelShortHyenaOperator):
        assert operator.hidden_size == 864
        assert operator.pregate
        assert operator.postgate
        num_weights = sum([p.numel() for p in operator.parameters()])
        assert num_weights == 6048

    def test_gpu_forward(self, operator: ParallelShortHyenaOperator):
        device = torch.device("cuda")
        operator = operator.to(device)
        batch_size = 2
        seq_len = 1024
        g = operator.num_groups
        dg = operator.group_dim

        x1 = torch.ones((batch_size, (g * dg), seq_len), device=device)
        x2 = torch.ones((batch_size, (g * dg), seq_len), device=device)
        v = torch.ones((batch_size, (g * dg), seq_len), device=device)

        output = operator(x1, x2, v)
        assert output.shape[0] == batch_size
        assert output.shape[1] == operator.hidden_size
        assert output.shape[2] == seq_len


class TestParallelShortHyenaOperatorWithConvBias:
    @pytest.fixture
    def operator(self, transformer_config: TransformerConfig, hyena_config: HyenaConfig) -> ParallelShortHyenaOperator:
        with simple_parallel_state():
            yield ParallelShortHyenaOperator(
                hidden_size=transformer_config.hidden_size,
                transformer_config=transformer_config,
                hyena_config=hyena_config,
                init_method="small_init",
                short_conv_class=ParallelCausalDepthwiseConv1d,
                use_fast_causal_conv=False,
                local_init=False,
                use_conv_bias=True,
            )

    def test_initialization(self, operator: ParallelShortHyenaOperator):
        assert operator.hidden_size == 864
        assert operator.pregate
        assert operator.postgate
        num_weights = sum([p.numel() for p in operator.parameters()])
        assert num_weights == 6912

    def test_gpu_forward(self, operator: ParallelShortHyenaOperator):
        device = torch.device("cuda")
        operator = operator.to(device)
        batch_size = 2
        seq_len = 1024
        g = operator.num_groups
        dg = operator.group_dim

        x1 = torch.ones((batch_size, (g * dg), seq_len), device=device)
        x2 = torch.ones((batch_size, (g * dg), seq_len), device=device)
        v = torch.ones((batch_size, (g * dg), seq_len), device=device)

        output = operator(x1, x2, v)
        assert output.shape[0] == batch_size
        assert output.shape[1] == operator.hidden_size
        assert output.shape[2] == seq_len


class TestParallelCausalDepthwiseConv1d:
    @pytest.fixture
    def operator(
        self, transformer_config: TransformerConfig, hyena_config: HyenaConfig
    ) -> ParallelCausalDepthwiseConv1d:
        with simple_parallel_state():
            yield ParallelCausalDepthwiseConv1d(
                d_model=transformer_config.hidden_size,
                transformer_config=transformer_config,
                hyena_config=hyena_config,
                kernel_size=hyena_config.short_conv_L,
                init_method=transformer_config.init_method,
                bias=hyena_config.conv_proj_bias,
                use_fast_causal_conv=hyena_config.fast_conv_proj,
            )

    def test_initialization(self, operator: ParallelCausalDepthwiseConv1d):
        assert operator.d_model == 864
        assert operator.kernel_size == 3
        assert operator.use_conv_bias
        num_weights = sum([p.numel() for p in operator.parameters()])
        assert num_weights == 2592

    def test_gpu_forward(self, operator: ParallelCausalDepthwiseConv1d):
        device = torch.device("cuda")
        operator = operator.to(device)
        batch_size = 2
        d_model = operator.d_model
        seq_len = 1024

        x1 = torch.ones((batch_size, d_model, seq_len), device=device)
        output = operator(x1, False)

        assert output.shape[0] == batch_size
        assert output.shape[1] == d_model
        assert output.shape[2] == seq_len