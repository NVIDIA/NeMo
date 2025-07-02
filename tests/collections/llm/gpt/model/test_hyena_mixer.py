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

import contextlib
import os

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from nemo.collections.llm.gpt.model.hyena import HyenaNVTestConfig, HyenaTestConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_layer_specs import hyena_stack_spec_no_te
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_mixer import HyenaMixer

# Add skip decorator for GPU tests
skip_if_no_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")


@contextlib.contextmanager
def init_distributed_parallel_state(
    world_size=1, rank=0, tensor_model_parallel_size=1, context_parallel_size=1, pipeline_model_parallel_size=1
):
    """Initialize a distributed environment for testing.

    Creates a real distributed environment with specified parameters.
    """
    # Initialize distributed with a single process
    if not dist.is_initialized():
        # Setup minimal environment for single process distributed
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Set device
        torch.cuda.set_device(0)

        # Initialize process group
        dist.init_process_group(backend="nccl")

    # Initialize model parallel
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
    )

    # Initialize the model parallel RNG
    model_parallel_cuda_manual_seed(42)

    try:
        yield
    finally:
        # Clean up
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture(params=[pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float32, id="fp32")])
def dtype(request):
    """Parametrized dtype fixture"""
    return request.param


@pytest.fixture(params=[pytest.param("standard", id="non_nv"), pytest.param("nv", id="nv")])
def config_type(request):
    """Parametrized config type fixture"""
    return request.param


@pytest.fixture
def test_config(dtype, config_type) -> HyenaTestConfig:
    """Create a test config based on the parametrized dtype and config type"""
    if config_type == "standard":
        config = HyenaTestConfig()
    else:  # nv
        config = HyenaNVTestConfig()

    config.params_dtype = dtype
    return config


@pytest.fixture
def hyena_config() -> HyenaConfig:
    config = HyenaConfig()
    config.num_groups_hyena = 4096
    config.num_groups_hyena_short = 256
    config.num_groups_hyena_medium = 256
    return config


@pytest.fixture(params=[pytest.param("hyena_short_conv", id="short"), pytest.param("hyena_medium_conv", id="medium")])
def operator_type(request):
    """Parametrized operator type fixture"""
    return request.param


@pytest.fixture
def hyena_mixer(test_config: HyenaTestConfig, hyena_config: HyenaConfig, operator_type: str):
    """Create a HyenaMixer instance for testing."""
    with init_distributed_parallel_state(world_size=1):
        # Create submodules
        submodules = hyena_stack_spec_no_te.submodules.hyena_layer.submodules.mixer.submodules

        # Create mixer
        mixer = HyenaMixer(
            transformer_config=test_config,
            hyena_config=hyena_config,
            max_sequence_length=512,
            submodules=submodules,
            layer_number=1,
            operator_type=operator_type,
        )
        yield mixer


@skip_if_no_gpu
def test_mixer_initialization(
    hyena_mixer: HyenaMixer, test_config: HyenaTestConfig, hyena_config: HyenaConfig, operator_type: str
):
    """Test proper initialization of HyenaMixer with different configurations."""
    with init_distributed_parallel_state(world_size=1):
        # Verify basic attributes
        assert hyena_mixer.transformer_config == test_config
        assert hyena_mixer.hyena_config == hyena_config
        assert hyena_mixer.operator_type == operator_type
        assert hyena_mixer.layer_number == 1

        # Verify model parallel attributes
        assert hyena_mixer.model_parallel_size == 1
        assert hyena_mixer.hidden_size_per_partition == hyena_mixer.hidden_size

        # Verify projection attributes
        assert hyena_mixer.proj_groups == hyena_config.proj_groups
        assert hyena_mixer.tie_projection_weights == hyena_config.tie_projection_weights

        # Verify mixer type based on operator_type
        if operator_type == "hyena_short_conv":
            assert hyena_mixer.num_groups == hyena_config.num_groups_hyena_short
        elif operator_type == "hyena_medium_conv":
            assert hyena_mixer.num_groups == hyena_config.num_groups_hyena_medium
        else:
            assert hyena_mixer.num_groups == hyena_config.num_groups_hyena


@skip_if_no_gpu
def test_mixer_forward_pass(hyena_mixer: HyenaMixer):
    """Test forward pass of HyenaMixer with different input shapes and configurations."""
    with init_distributed_parallel_state(world_size=1):
        # Test different batch sizes and sequence lengths
        test_cases = [
            (1, 128),  # Small batch, short sequence
            (2, 512),  # Medium batch, medium sequence
            (4, 1024),  # Large batch, long sequence
        ]

        for batch_size, seq_len in test_cases:
            # Create input tensor
            input_features = torch.rand(
                (seq_len, batch_size, hyena_mixer.hidden_size),
                dtype=hyena_mixer.transformer_config.params_dtype,
                device=torch.cuda.current_device(),
            )

            # Forward pass
            y, bias = hyena_mixer(input_features, _hyena_use_cp=False)

            # Verify output shape
            expected_shape = (seq_len, batch_size, hyena_mixer.hidden_size)
            assert y.shape == expected_shape, f"Expected shape {expected_shape}, got {y.shape}"

            # Verify output is not NaN
            assert not torch.isnan(y).any(), "Output contains NaN values"
            # Verify output is not Inf
            assert not torch.isinf(y).any(), "Output contains Inf values"


@skip_if_no_gpu
def test_mixer_dtypes(hyena_mixer: HyenaMixer, dtype: torch.dtype):
    """Test HyenaMixer with different input data types."""
    with init_distributed_parallel_state(world_size=1):
        batch_size = 2
        seq_len = 512

        input_features = torch.rand(
            (seq_len, batch_size, hyena_mixer.hidden_size),
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

        # Forward pass
        y, bias = hyena_mixer(input_features, _hyena_use_cp=False)

        # Verify output dtype matches input dtype
        assert y.dtype == dtype, f"Expected output dtype {dtype}, got {y.dtype}"
        assert bias.dtype == dtype, f"Expected bias dtype {dtype}, got {bias.dtype}"


@skip_if_no_gpu
def test_mixer_state_dict(hyena_mixer: HyenaMixer, operator_type: str):
    """Test state dict functionality of HyenaMixer."""
    with init_distributed_parallel_state(world_size=1):
        # Get state dict
        state_dict = hyena_mixer.state_dict()

        # Create new mixer with same config
        submodules = hyena_stack_spec_no_te.submodules.hyena_layer.submodules.mixer.submodules
        new_mixer = HyenaMixer(
            transformer_config=hyena_mixer.transformer_config,
            hyena_config=hyena_mixer.hyena_config,
            max_sequence_length=512,
            submodules=submodules,
            layer_number=1,
            operator_type=operator_type,
        )

        # Load state dict
        new_mixer.load_state_dict(state_dict)

        # Verify parameters match
        for (name1, param1), (name2, param2) in zip(hyena_mixer.named_parameters(), new_mixer.named_parameters()):
            assert torch.allclose(param1, param2), f"Parameter mismatch after loading state dict: {name1}"
