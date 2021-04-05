# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import pytest
import torch
from numba import cuda

from nemo.collections.asr.parts.numba import __NUMBA_MINIMUM_VERSION__, numba_utils
from nemo.collections.asr.parts.numba.rnnt_loss import rnnt_numpy
from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import certify_inputs
from nemo.collections.asr.parts.numba.rnnt_loss.utils.cuda_utils import gpu_rnnt_kernel, reduce


def log_softmax(x, axis=-1):
    x = torch.from_numpy(x)  # zero-copy
    x = torch.log_softmax(x, axis)
    x = x.numpy()
    return x


class TestRNNTCUDAKernels:
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Reductions can only be run when CUDA is available")
    @pytest.mark.unit
    def test_compute_alphas_kernel(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        random = np.random.RandomState(0)
        original_shape = [1, 5, 11, 3]
        B, T, U, V = original_shape

        # Numpy kernel
        x = random.randn(*original_shape)
        labels = np.array([[1, 1, 1, 2, 2, 2, 1, 2, 2, 1]])  # [1, 10]
        label_len = len(labels[0]) + 1
        blank_idx = 0

        x_np = log_softmax(x, axis=-1)
        ground_alphas, ground_log_likelihood = rnnt_numpy.forward_pass(
            x_np[0, :, :label_len, :], labels[0, : label_len - 1], blank_idx
        )

        # Pytorch kernel
        device = torch.device('cuda')
        if hasattr(cuda, 'external_stream'):
            stream = cuda.external_stream(torch.cuda.current_stream(device).cuda_stream)
        else:
            stream = cuda.default_stream()

        x_c = torch.tensor(x, device=device, dtype=torch.float32)
        labels_c = torch.tensor(labels, device=device, dtype=torch.int32)

        # Allocate workspace memory
        denom = torch.zeros(B * T * U, device=device, dtype=x_c.dtype)
        alphas = torch.zeros(B * T * U, device=device, dtype=x_c.dtype)
        llForward = torch.zeros(B, device=device, dtype=x_c.dtype)
        input_lengths = torch.tensor([T], dtype=torch.int32, device=device)
        label_lengths = torch.tensor([len(labels[0])], dtype=torch.int32, device=device)

        # certify input data
        certify_inputs(x_c, labels_c, input_lengths, label_lengths)

        # flatten activation tensor (for pointer based indexing)
        x_c = x_c.view([-1])

        # call kernel
        # log softmax reduction
        reduce.reduce_max(x_c, denom, rows=V, cols=B * T * U, minus=False, stream=stream)
        reduce.reduce_exp(x_c, denom, rows=V, cols=B * T * U, minus=True, stream=stream)

        # alpha kernel
        gpu_rnnt_kernel.compute_alphas_kernel[B, U, stream, 0](
            x_c, denom, alphas, llForward, input_lengths, label_lengths, labels_c, B, T, U, V, blank_idx,
        )

        # sync kernel
        stream.synchronize()

        # reshape alphas
        alphas = alphas.view([B, T, U])
        diff = ground_alphas - alphas[0].cpu().numpy()

        assert np.abs(diff).mean() <= 1e-5
        assert np.square(diff).mean() <= 1e-10

        ll_diff = ground_log_likelihood - llForward[0].cpu().numpy()

        assert np.abs(ll_diff).mean() <= 1e-5
        assert np.square(ll_diff).mean() <= 1e-10

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Reductions can only be run when CUDA is available")
    @pytest.mark.unit
    def test_compute_betas_kernel(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        random = np.random.RandomState(0)
        original_shape = [1, 5, 11, 3]
        B, T, U, V = original_shape

        # Numpy kernel
        x = random.randn(*original_shape)
        labels = np.array([[1, 1, 1, 2, 2, 2, 1, 2, 2, 1]])  # [1, 10]
        label_len = len(labels[0]) + 1
        blank_idx = 0

        x_np = log_softmax(x, axis=-1)
        ground_alphas, ground_log_likelihood = rnnt_numpy.backward_pass(
            x_np[0, :, :label_len, :], labels[0, : label_len - 1], blank_idx
        )

        # Pytorch kernel
        device = torch.device('cuda')
        if hasattr(cuda, 'external_stream'):
            stream = cuda.external_stream(torch.cuda.current_stream(device).cuda_stream)
        else:
            stream = cuda.default_stream()

        x_c = torch.tensor(x, device=device, dtype=torch.float32)
        labels_c = torch.tensor(labels, device=device, dtype=torch.int32)

        # Allocate workspace memory
        denom = torch.zeros(B * T * U, device=device, dtype=x_c.dtype)
        betas = torch.zeros(B * T * U, device=device, dtype=x_c.dtype)
        llBackward = torch.zeros(B, device=device, dtype=x_c.dtype)
        input_lengths = torch.tensor([T], dtype=torch.int32, device=device)
        label_lengths = torch.tensor([len(labels[0])], dtype=torch.int32, device=device)

        # certify input data
        certify_inputs(x_c, labels_c, input_lengths, label_lengths)

        # flatten activation tensor (for pointer based indexing)
        x_c = x_c.view([-1])

        # call kernel
        # log softmax reduction
        reduce.reduce_max(x_c, denom, rows=V, cols=B * T * U, minus=False, stream=stream)
        reduce.reduce_exp(x_c, denom, rows=V, cols=B * T * U, minus=True, stream=stream)

        # beta kernel
        gpu_rnnt_kernel.compute_betas_kernel[B, U, stream, 0](
            x_c, denom, betas, llBackward, input_lengths, label_lengths, labels_c, B, T, U, V, blank_idx,
        )

        # sync kernel
        stream.synchronize()

        # reshape alphas
        betas = betas.view([B, T, U])
        diff = ground_alphas - betas[0].cpu().numpy()

        assert np.abs(diff).mean() <= 1e-5
        assert np.square(diff).mean() <= 1e-10

        ll_diff = ground_log_likelihood - llBackward[0].cpu().numpy()

        assert np.abs(ll_diff).mean() <= 1e-5
        assert np.square(ll_diff).mean() <= 1e-10
