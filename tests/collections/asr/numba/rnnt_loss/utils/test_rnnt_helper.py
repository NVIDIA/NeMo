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
from numba import cuda

from nemo.collections.asr.parts.numba import __NUMBA_MINIMUM_VERSION__, numba_utils
from nemo.collections.asr.parts.numba.rnnt_loss.utils import global_constants, rnnt_helper


class TestRNNTHelper:
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Helpers can only be run when CUDA is available")
    @pytest.mark.unit
    def test_log_sum_exp(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        # wrapper kernel for device function that is tested
        @cuda.jit
        def _kernel(x, y):
            x_pos = cuda.grid(1)
            if x_pos < x.shape[0] and x_pos < y.shape[0]:
                x[x_pos] = rnnt_helper.log_sum_exp(x[x_pos], y[x_pos])

        x = np.zeros([8])  # np.random.rand(8192)
        y = np.ones([8])  # np.random.rand(8192)

        stream = cuda.stream()
        x_c = cuda.to_device(x, stream=stream)
        y_c = cuda.to_device(y, stream=stream)

        # call kernel
        threads_per_block = global_constants.threads_per_block()
        blocks_per_grid = (x.shape[0] + threads_per_block - 1) // threads_per_block
        _kernel[blocks_per_grid, threads_per_block, stream](x_c, y_c)

        # sync kernel
        stream.synchronize()

        x_new = x_c.copy_to_host(stream=stream)
        del x_c, y_c

        assert (x_new.sum() - 10.506093500145782) <= 1e-5

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Helpers can only be run when CUDA is available")
    @pytest.mark.unit
    def test_log_sum_exp_neg_inf(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        # wrapper kernel for device function that is tested
        @cuda.jit
        def _kernel(x, y):
            x_pos = cuda.grid(1)
            if x_pos < x.shape[0] and x_pos < y.shape[0]:
                x[x_pos] = rnnt_helper.log_sum_exp(x[x_pos], y[x_pos])

        x = np.asarray([global_constants.FP32_NEG_INF] * 8)
        y = np.ones([len(x)])

        stream = cuda.stream()
        x_c = cuda.to_device(x, stream=stream)
        y_c = cuda.to_device(y, stream=stream)

        # call kernel
        threads_per_block = global_constants.threads_per_block()
        blocks_per_grid = (x.shape[0] + threads_per_block - 1) // threads_per_block
        _kernel[blocks_per_grid, threads_per_block, stream](x_c, y_c)

        # sync kernel
        stream.synchronize()

        x_new = x_c.copy_to_host(stream=stream)
        del x_c, y_c

        assert np.allclose(x_new, np.ones_like(x_new), atol=1e-5)

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Helpers can only be run when CUDA is available")
    @pytest.mark.unit
    def test_div_up(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        # wrapper kernel for device function that is tested
        @cuda.jit
        def _kernel(x, y):
            x_pos = cuda.grid(1)
            if x_pos < x.shape[0] and x_pos < y.shape[0]:
                x[x_pos] = rnnt_helper.div_up(x[x_pos], y[x_pos])

        x = np.full([8], fill_value=10)  # np.random.rand(8192)
        y = np.full([8], fill_value=2)  # np.random.rand(8192)

        stream = cuda.stream()
        x_c = cuda.to_device(x, stream=stream)
        y_c = cuda.to_device(y, stream=stream)

        # call kernel
        threads_per_block = global_constants.threads_per_block()
        blocks_per_grid = (x.shape[0] + threads_per_block - 1) // threads_per_block
        _kernel[blocks_per_grid, threads_per_block, stream](x_c, y_c)

        # sync kernel
        stream.synchronize()

        x_new = x_c.copy_to_host(stream=stream)
        del x_c, y_c

        for i in range(len(x_new)):
            assert x_new[i] == ((10 + 2 - 1) // 2)

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Helpers can only be run when CUDA is available")
    @pytest.mark.unit
    def test_add(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        # wrapper kernel for device function that is tested
        @cuda.jit
        def _kernel(x, y):
            x_pos = cuda.grid(1)
            if x_pos < x.shape[0] and x_pos < y.shape[0]:
                x[x_pos] = rnnt_helper.add(x[x_pos], y[x_pos])

        x = np.full([8], fill_value=10)  # np.random.rand(8192)
        y = np.full([8], fill_value=2)  # np.random.rand(8192)

        stream = cuda.stream()
        x_c = cuda.to_device(x, stream=stream)
        y_c = cuda.to_device(y, stream=stream)

        # call kernel
        threads_per_block = global_constants.threads_per_block()
        blocks_per_grid = (x.shape[0] + threads_per_block - 1) // threads_per_block
        _kernel[blocks_per_grid, threads_per_block, stream](x_c, y_c)

        # sync kernel
        stream.synchronize()

        x_new = x_c.copy_to_host(stream=stream)
        del x_c, y_c

        for i in range(len(x_new)):
            assert x_new[i] == 12

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Helpers can only be run when CUDA is available")
    @pytest.mark.unit
    def test_maximum(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        # wrapper kernel for device function that is tested
        @cuda.jit
        def _kernel(x, y):
            x_pos = cuda.grid(1)
            if x_pos < x.shape[0] and x_pos < y.shape[0]:
                x[x_pos] = rnnt_helper.maximum(x[x_pos], y[x_pos])

        x = np.full([8], fill_value=10)  # np.random.rand(8192)
        y = np.full([8], fill_value=2)  # np.random.rand(8192)

        stream = cuda.stream()
        x_c = cuda.to_device(x, stream=stream)
        y_c = cuda.to_device(y, stream=stream)

        # call kernel
        threads_per_block = global_constants.threads_per_block()
        blocks_per_grid = (x.shape[0] + threads_per_block - 1) // threads_per_block
        _kernel[blocks_per_grid, threads_per_block, stream](x_c, y_c)

        # sync kernel
        stream.synchronize()

        x_new = x_c.copy_to_host(stream=stream)
        del x_c, y_c

        for i in range(len(x_new)):
            assert x_new[i] == 10

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Helpers can only be run when CUDA is available")
    @pytest.mark.unit
    def test_identity(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        # wrapper kernel for device function that is tested
        @cuda.jit
        def _kernel(x):
            x_pos = cuda.grid(1)
            if x_pos < x.shape[0]:
                x[x_pos] = rnnt_helper.identity(x[x_pos])

        x = np.full([8], fill_value=10)  # np.random.rand(8192)

        stream = cuda.stream()
        x_c = cuda.to_device(x, stream=stream)

        # call kernel
        threads_per_block = global_constants.threads_per_block()
        blocks_per_grid = (x.shape[0] + threads_per_block - 1) // threads_per_block
        _kernel[blocks_per_grid, threads_per_block, stream](x_c)

        # sync kernel
        stream.synchronize()

        x_new = x_c.copy_to_host(stream=stream)
        del x_c

        for i in range(len(x_new)):
            assert x_new[i] == x[i]

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Helpers can only be run when CUDA is available")
    @pytest.mark.unit
    def test_negate(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        # wrapper kernel for device function that is tested
        @cuda.jit
        def _kernel(x):
            x_pos = cuda.grid(1)
            if x_pos < x.shape[0]:
                x[x_pos] = rnnt_helper.negate(x[x_pos])

        x = np.full([8], fill_value=10)  # np.random.rand(8192)

        stream = cuda.stream()
        x_c = cuda.to_device(x, stream=stream)

        # call kernel
        threads_per_block = global_constants.threads_per_block()
        blocks_per_grid = (x.shape[0] + threads_per_block - 1) // threads_per_block
        _kernel[blocks_per_grid, threads_per_block, stream](x_c)

        # sync kernel
        stream.synchronize()

        x_new = x_c.copy_to_host(stream=stream)
        del x_c

        for i in range(len(x_new)):
            assert x_new[i] == -x[i]

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Helpers can only be run when CUDA is available")
    @pytest.mark.unit
    def test_exponential(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        # wrapper kernel for device function that is tested
        @cuda.jit
        def _kernel(x):
            x_pos = cuda.grid(1)
            if x_pos < x.shape[0]:
                x[x_pos] = rnnt_helper.exponential(x[x_pos])

        x = np.random.rand(8)

        stream = cuda.stream()
        x_c = cuda.to_device(x, stream=stream)

        # call kernel
        threads_per_block = global_constants.threads_per_block()
        blocks_per_grid = (x.shape[0] + threads_per_block - 1) // threads_per_block
        _kernel[blocks_per_grid, threads_per_block, stream](x_c)

        # sync kernel
        stream.synchronize()

        x_new = x_c.copy_to_host(stream=stream)
        del x_c

        y = np.exp(x)
        for i in range(len(x_new)):
            assert (x_new[i] - y[i]) < 1e-4

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Helpers can only be run when CUDA is available")
    @pytest.mark.unit
    def test_log_plus(self):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        # wrapper kernel for device function that is tested
        @cuda.jit
        def _kernel(x, y):
            x_pos = cuda.grid(1)
            if x_pos < x.shape[0] and x_pos < y.shape[0]:
                x[x_pos] = rnnt_helper.log_plus(x[x_pos], y[x_pos])

        x = np.full([8], fill_value=10.0)  # np.random.rand(8192)
        y = np.full([8], fill_value=2.0)  # np.random.rand(8192)

        stream = cuda.stream()
        x_c = cuda.to_device(x, stream=stream)
        y_c = cuda.to_device(y, stream=stream)

        # call kernel
        threads_per_block = global_constants.threads_per_block()
        blocks_per_grid = (x.shape[0] + threads_per_block - 1) // threads_per_block
        _kernel[blocks_per_grid, threads_per_block, stream](x_c, y_c)

        # sync kernel
        stream.synchronize()

        x_new = x_c.copy_to_host(stream=stream)
        del x_c, y_c

        z = np.log1p(np.exp(-np.fabs(x - y))) + np.maximum(x, y)

        for i in range(len(x_new)):
            assert x_new[i] == z[i]


if __name__ == '__main__':
    pytest.main([__file__])
