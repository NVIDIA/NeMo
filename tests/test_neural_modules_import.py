# ! /usr/bin/python
# -*- coding: utf-8 -*-

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

import yaml

import nemo
from tests.common_setup import NeMoUnitTest


class NeuralModuleImportTest(NeMoUnitTest):
    """
        Class testing Neural Module configuration export.
    """

    class MockupSimpleModule(nemo.core.NeuralModule):
        """
        Mockup component class.
        """

        def __init__(self, a, b, c, d):
            super().__init__()

    def setUp(self) -> None:
        super().setUp()

        # Mockup abstract methods.
        NeuralModuleImportTest.MockupSimpleModule.__abstractmethods__ = set()

    def test_simple_export(self):
        """ Tests whether build-in types are properly exported."""

        # params = {"int": 123, "float": 12.4, "string": "ala ma kota", "bool": True}
        orig_module = NeuralModuleImportTest.MockupSimpleModule(123, 12.4, "ala ma kota", True)

        # Export.
        orig_module.export_config("simple_export.yml", "/tmp/")

        # Import and create the new object.
        new_module = NeuralModuleImportTest.MockupSimpleModule.import_config("simple_export.yml", "/tmp/")

        # Compare class types.
        self.assertEqual(type(orig_module).__name__, type(new_module).__name__)

        # Compare objects - by its all params.
        param_keys = orig_module.init_params.keys()
        for key in param_keys:
            self.assertEqual(orig_module.init_params[key], new_module.init_params[key])
