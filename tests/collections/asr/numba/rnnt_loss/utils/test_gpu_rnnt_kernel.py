import pytest
import numpy as np
from numba import cuda

from nemo.collections.asr.parts.numba.rnnt_loss.utils.cuda_utils import gpu_rnnt_kernel
from nemo.collections.asr.parts.numba.rnnt_loss import rnnt_numpy
from nemo.collections.asr.parts.numba.rnnt_loss.utils import global_constants


class TestRNNTCUDAKernels:

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA Reductions can only be run when CUDA is available")
    @pytest.mark.unit
    def test_compute_alphas_kernel(self):
        random = np.random.RandomState(0)
        original_shape = [1, 5, 4, 3]
        x = random.randn(*original_shape).reshape([-1])
        dx = random.randn(*x.shape)

        # stream = cuda.stream()
        # x_c = cuda.to_device(x, stream=stream)
        # dx_c = cuda.to_device(dx, stream=stream)
        #
        # # call kernel
        # cols = np.prod(original_shape[:3])
        # reduce.reduce_max(x_c, dx_c, rows=original_shape[-1], cols=cols, minus=False, stream=stream)
        #
        # # sync kernel
        # stream.synchronize()
        #
        # dx_result = dx_c.copy_to_host(stream=stream)
        # del x_c, dx_c
        #
        # # collect results in first [B * T * U] values; for all V
        # assert np.abs(dx_result[cols:] - dx[cols:]).sum() <= 1e-7
        # # make sure dx_result updates the [B * T * U] values
        # assert np.abs(dx_result[:cols] - dx[:cols]).sum() > 0

