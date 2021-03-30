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

import operator

from nemo.utils import model_utils


def numba_cuda_is_supported(version) -> bool:
    module_available, msg = model_utils.check_lib_version('numba', checked_version=version, operator=operator.ge)

    # If numba is not installed
    if module_available is None:
        return False

    # If numba version is installed and available
    if module_available is True:
        from numba import cuda

        # this method first arrived in 0.53, and that's the minimum version required
        if hasattr(cuda, 'is_supported_version'):
            return cuda.is_supported_version()
        else:
            # assume cuda is supported, but it may fail due to CUDA incompatibility
            return False

    else:
        return False


def skip_numba_cuda_test_if_unsupported(version):
    numba_cuda_support = numba_cuda_is_supported(version)
    if not numba_cuda_support:
        import pytest

        pytest.skip(f"Numba cuda test is being skipped. Minimum version required : {version}")
