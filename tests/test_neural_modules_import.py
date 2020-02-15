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

import nemo
from tests.common_setup import NeMoUnitTest


class NeuralModuleImportTest(NeMoUnitTest):
    """
        Class testing Neural Module configuration export.
    """

    class FirstSimpleModule(nemo.core.NeuralModule):
        """
        Mockup component class.
        """

        def __init__(self, a, b, c, d):
            super().__init__()

    class SecondSimpleModule(nemo.core.NeuralModule):
        """
        Mockup component class.
        """

        def __init__(self, x, y):
            super().__init__()

    def setUp(self) -> None:
        super().setUp()

        # Mockup abstract methods.
        NeuralModuleImportTest.FirstSimpleModule.__abstractmethods__ = set()
        NeuralModuleImportTest.SecondSimpleModule.__abstractmethods__ = set()

    def test_simple_import_root_neural_module(self):
        """ Tests whether the Neural Module can instantiate a simple module by loading a configuration file."""

        # params = {"int": 123, "float": 12.4, "string": "ala ma kota", "bool": True}
        orig_module = NeuralModuleImportTest.FirstSimpleModule(123, 12.4, "ala ma kota", True)

        # Export.
        orig_module.export_to_config("/tmp/first_simple_import.yml")

        # Import and create the new object.
        new_module = nemo.core.NeuralModule.import_from_config("/tmp/first_simple_import.yml")

        # Compare class types.
        self.assertEqual(type(orig_module).__name__, type(new_module).__name__)

        # Compare objects - by its all params.
        param_keys = orig_module.init_params.keys()
        for key in param_keys:
            self.assertEqual(orig_module.init_params[key], new_module.init_params[key])

    def test_simple_import_leaf_module(self):
        """
            Tests whether a particular module can instantiate another
            instance (a copy) by loading a configuration file.
        """

        # params = {"int": 123, "float": 12.4, "string": "ala ma kota", "bool": True}
        orig_module = NeuralModuleImportTest.FirstSimpleModule(123, 12.4, "ala ma kota", True)

        # Export.
        orig_module.export_to_config("/tmp/first_simple_import.yml")

        # Import and create the new object.
        new_module = NeuralModuleImportTest.FirstSimpleModule.import_from_config("/tmp/first_simple_import.yml")

        # Compare class types.
        self.assertEqual(type(orig_module).__name__, type(new_module).__name__)

        # Compare objects - by its all params.
        param_keys = orig_module.init_params.keys()
        for key in param_keys:
            self.assertEqual(orig_module.init_params[key], new_module.init_params[key])

    def test_incompatible_import_leaf_module(self):
        """
            Tests whether a particular module can instantiate another
            instance (a copy) by loading a configuration file.
        """

        # params = {"int": 123, "float": 12.4, "string": "ala ma kota", "bool": True}
        orig_module = NeuralModuleImportTest.SecondSimpleModule(["No", "way", "dude!"], None)

        # Export.
        orig_module.export_to_config("/tmp/second_simple_import.yml")

        # This will actuall create an instance of SecondSimpleModule - OK.
        new_module = nemo.core.NeuralModule.import_from_config("/tmp/second_simple_import.yml")
        # Compare class types.
        self.assertEqual(type(orig_module).__name__, type(new_module).__name__)

        # This will create an instance of SecondSimpleModule, not FirstSimpleModule - SO NOT OK!!
        with self.assertRaises(ImportError):
            _ = NeuralModuleImportTest.FirstSimpleModule.import_from_config("/tmp/second_simple_import.yml")
