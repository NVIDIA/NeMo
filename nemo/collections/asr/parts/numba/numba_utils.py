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


def numba_cuda_is_supported(min_version: str) -> bool:
    """
    Tests if an appropriate version of numba is installed, and if it is,
    if cuda is supported properly within it.
    
    Args:
        min_version: The minimum version of numba that is required.

    Returns:
        bool, whether cuda is supported with this current installation or not.
    """
    module_available, msg = model_utils.check_lib_version('numba', checked_version=min_version, operator=operator.ge)

    # If numba is not installed
    if module_available is None:
        return False

    # If numba version is installed and available
    if module_available is True:
        from numba import cuda

        # this method first arrived in 0.53, and that's the minimum version required
        if hasattr(cuda, 'is_supported_version'):
            try:
                return cuda.is_available() and cuda.is_supported_version()
            except OSError:
                # dlopen(libcudart.dylib) might fail if CUDA was never installed in the first place.
                return False
        else:
            # assume cuda is supported, but it may fail due to CUDA incompatibility
            return False

    else:
        return False


def skip_numba_cuda_test_if_unsupported(min_version: str):
    """
    Helper method to skip pytest test case if numba cuda is not supported.
    
    Args:
        min_version: The minimum version of numba that is required.
    """
    numba_cuda_support = numba_cuda_is_supported(min_version)
    if not numba_cuda_support:
        import pytest

        pytest.skip(f"Numba cuda test is being skipped. Minimum version required : {min_version}")
