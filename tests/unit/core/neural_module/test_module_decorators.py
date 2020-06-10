# ! /usr/bin/python
# -*- coding: utf-8 -*-

# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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

from nemo.core import NeuralModule, DeviceType, skip_in_data_parallel, run_only_on_device

@pytest.mark.usefixtures("neural_factory")
class TestModuleDecorators:
    """
        Class testing Neural Module configuration export.
    """

    @skip_in_data_parallel
    class MockupModuleWithSkip(NeuralModule):
        """
        Mockup component class.
        """

        def __init__(self):
            super().__init__()

    @run_only_on_device(device_type=DeviceType.CPU)    
    class MockupModuleOnCPU(NeuralModule):
        """
        Mockup component class.
        """

        def __init__(self):
            super().__init__()

    def setup_method(self, method):
        """ 
            Setup_method is invoked for every test method of a class.
            Mocks up the module class.
        """
        # Mockup abstract methods.
        TestModuleDecorators.MockupModuleWithSkip.__abstractmethods__ = set()
        TestModuleDecorators.MockupModuleOnCPU.__abstractmethods__ = set()

    @pytest.mark.unit
    def test_module_decorators(self):
        """
            Tests whether skip dataparallel decorator works.
        """
        # Create a module that has skip by definition.
        module1 = TestModuleDecorators.MockupModuleWithSkip()
        assert module1.skip_in_data_parallel == True
        assert hasattr(module1, "run_only_on_device") == False

        # Create a module that has no skip by definition.
        module2 = TestModuleDecorators.MockupModuleOnCPU()
        assert hasattr(module2, "skip_in_data_parallel") == False
        assert module2.run_only_on_device == DeviceType.CPU
