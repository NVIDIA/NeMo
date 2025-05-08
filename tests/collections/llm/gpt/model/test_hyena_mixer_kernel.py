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
from einops import rearrange
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from nemo.collections.llm.gpt.model.hyena import HyenaNVTestConfig, HyenaTestConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_layer_specs import hyena_stack_spec_no_te
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_mixer import HyenaMixer


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


class B2BConv1d(torch.nn.Module):
    def __init__(self, hyena_config, hyena_test_config, seq_len, use_b2b_causal_conv1d=False):
        super().__init__()

        # Create necessary submodules - use the mixer submodules like in the regular mixer fixture
        submodules = hyena_stack_spec_no_te.submodules.hyena_layer.submodules.mixer.submodules

        # Set the b2b parameter in the config
        hyena_test_config.use_b2b_causal_conv1d = use_b2b_causal_conv1d

        print("Creating HyenaMixer...")
        self.mixer = HyenaMixer(
            transformer_config=hyena_test_config,
            hyena_config=hyena_config,
            max_sequence_length=seq_len,
            submodules=submodules,
            layer_number=1,
            operator_type="hyena_short_conv",
        )

    def forward(self, x, _use_cp=True):
        features = self.mixer.hyena_proj_conv(x, _use_cp=_use_cp)
        x1, x2, v = rearrange(
            features, "b (g dg p) l -> b (g dg) p l", p=3, g=self.mixer.num_groups_per_tp_rank
        ).unbind(dim=2)
        z = self.mixer.mixer(x1, x2, v, _hyena_use_cp=_use_cp)
        return z


@pytest.fixture
def mixer(test_config: HyenaTestConfig, hyena_config: HyenaConfig):
    """Create a HyenaMixer instance for testing with PyTorch implementation"""
    with init_distributed_parallel_state(world_size=1):
        # Create the mixer
        mixer = B2BConv1d(hyena_config, test_config, seq_len=512, use_b2b_causal_conv1d=False)
        yield mixer


@pytest.fixture
def mixer_kernel(test_config: HyenaTestConfig, hyena_config: HyenaConfig):
    """Create a HyenaMixer instance for testing with CUDA kernel implementation"""
    with init_distributed_parallel_state(world_size=1):
        # Create the mixer
        mixer_kernel = B2BConv1d(hyena_config, test_config, seq_len=512, use_b2b_causal_conv1d=True)
        yield mixer_kernel


def test_b2b_causal_conv1d(mixer: B2BConv1d, mixer_kernel: B2BConv1d, config_type):
    # Skip NV config with CUDA kernel as it's not supported yet
    if config_type == "nv":
        # NV config may have conv bias which is not supported by CUDA kernel
        if isinstance(mixer_kernel.mixer.transformer_config, HyenaNVTestConfig):
            pytest.skip("NV config is not fully supported by b2b CUDA kernel yet")

    with init_distributed_parallel_state(world_size=1):
        batch_size = 2
        seq_len = 512
        input_features = torch.rand(
            (batch_size, mixer.mixer.hidden_size * 3, seq_len),
            dtype=mixer.mixer.transformer_config.params_dtype,
            device=torch.cuda.current_device(),
        )

        # PyTorch Mixer
        output_features = mixer(input_features, _use_cp=False)
        assert output_features.shape == (batch_size, mixer.mixer.hidden_size, seq_len), (
            f"output_features.shape: {output_features.shape}, batch_size: {batch_size}, mixer.mixer.hidden_size: {mixer.mixer.hidden_size}, seq_len: {seq_len}"
        )

        loss = output_features.float().mean()
        loss.backward()

        # Store the gradients for later comparison.
        grads = []
        for n, p in mixer.named_parameters():
            if p.grad is not None:
                grads.append((n, p.grad.clone()))

        mixer.zero_grad()

        # CUDA kernel in Mixer
        output_features_kernel = mixer_kernel(input_features, _use_cp=True)
        assert output_features_kernel.shape == (
            batch_size,
            mixer_kernel.mixer.hidden_size,
            seq_len,
        ), (
            f"output_features_kernel.shape: {output_features_kernel.shape}, batch_size: {batch_size}, mixer_kernel.mixer.hidden_size: {mixer_kernel.mixer.hidden_size}, seq_len: {seq_len}"
        )

        loss_kernel = output_features_kernel.float().mean()
        loss_kernel.backward()

        # Store the gradients for later comparison.
        grads_kernel = []
        for n, p in mixer_kernel.named_parameters():
            if p.grad is not None:
                grads_kernel.append((n, p.grad.clone()))

        mixer_kernel.zero_grad()

        # Compare results between PyTorch and CUDA kernel implementations
        torch.testing.assert_close(loss, loss_kernel)
        torch.testing.assert_close(output_features, output_features_kernel)

        # Compare gradients
        assert len(grads) == len(grads_kernel)

        gradient_mismatch = False
        for (n, g), (n_kernel, g_kernel) in zip(grads, grads_kernel):
            try:
                torch.testing.assert_close(g, g_kernel)
            except AssertionError as e:
                gradient_mismatch = True
                print(f"Gradient mismatch for {n}: {e}")

        if gradient_mismatch:
            print("There were gradient mismatches!")
        else:
            print("All gradients matched successfully!")
