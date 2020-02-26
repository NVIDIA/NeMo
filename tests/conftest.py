# =============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
# =============================================================================

import pytest

from nemo import logging
from nemo.core import DeviceType, NeuralModuleFactory


def pytest_addoption(parser):
    """ Additional command-line arguments passed to pytest. For now: --cpu """
    parser.addoption('--cpu', action='store_true', help="pass that argument to use CPU during testing (default: GPU)")


@pytest.fixture(scope="class")
def neural_factory(request):
    """ Fixture creating a Neural Factory object parametrized by the command line --cpu argument. """
    # Get flag.
    if request.config.getoption("--cpu"):
        device = DeviceType.CPU
    else:
        device = DeviceType.GPU
    # Initialize the default Neural Factory - on GPU.
    request.cls.nf = NeuralModuleFactory(placement=device)

    # Print standard header.
    logging.info("Using {} during testing".format(request.cls.nf.placement))
