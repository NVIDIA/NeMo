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


@pytest.fixture
def device(request):
    """ Simple fixture returning string denoting the device [CPU | GPU] """
    if request.config.getoption("--cpu"):
        return "CPU"
    else:
        return "GPU"


@pytest.fixture(scope="class")
def neural_factory(request):
    """ Fixture creating a Neural Factory object parametrized by the command line --cpu argument """
    # Get flag.
    if request.config.getoption("--cpu"):
        device = DeviceType.CPU
    else:
        device = DeviceType.GPU
    # Initialize the default Neural Factory - on GPU.
    request.cls.nf = NeuralModuleFactory(placement=device)

    # Print standard header.
    logging.info("Using {} during testing".format(request.cls.nf.placement))


@pytest.fixture(autouse=True)
def run_only_on_device_fixture(request, device):
    if request.node.get_closest_marker('run_only_on'):
        if request.node.get_closest_marker('run_only_on').args[0] != device:
            pytest.skip('skipped on this device: {}'.format(device))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "run_only_on(device): runs the test only on a given device [CPU | GPU]",
    )
