import pytest
import numpy as np
from numba import cuda

from nemo.collections.asr.parts.numba.rnnt_loss.utils.cuda_utils import reduce


class TestRNNTCUDAReductions:

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Reductions can only be run when CUDA is available")
    @pytest.mark.unit
    def test_reduce_max(self):
        random = np.random.RandomState(0)
        original_shape = [1, 5, 4, 2]
        x = random.randn(*original_shape).reshape([-1])
        dx = np.zeros_like(x)

        stream = cuda.stream()
        x_c = cuda.to_device(x, stream=stream)
        dx_c = cuda.to_device(dx, stream=stream)

        # call kernel
        reduce.reduce_max(x_c, dx_c, rows=10, cols=1 * 25 * 10, minus=False, stream=stream)

        # sync kernel
        stream.synchronize()

        dx = dx_c.copy_to_host(stream=stream)
        dx = dx.reshape(original_shape)
        del x_c, dx_c

        print(dx[0, 0, :])
        assert dx.shape == tuple(original_shape)

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Reductions can only be run when CUDA is available")
    @pytest.mark.unit
    def test_reduce_exp(self):
        random = np.random.RandomState(0)
        original_shape = [1, 5, 4, 2]
        x = random.randn(*original_shape).reshape([-1])
        dx = np.zeros_like(x)

        stream = cuda.stream()
        x_c = cuda.to_device(x, stream=stream)
        dx_c = cuda.to_device(dx, stream=stream)

        # call kernel
        reduce.reduce_exp(x_c, dx_c, rows=10, cols=1 * 25 * 10, minus=False, stream=stream)

        # sync kernel
        stream.synchronize()

        dx = dx_c.copy_to_host(stream=stream)
        dx = dx.reshape(original_shape)
        del x_c, dx_c

        print(dx[0, 0, :])
        assert dx.shape == tuple(original_shape)


if __name__ == '__main__':
    pytest.main([__file__])
