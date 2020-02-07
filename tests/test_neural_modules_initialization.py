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


class NeuralModuleConfigTest(NeMoUnitTest):
    """
        Class testing methods related to Neural Module import/export.
    """

    class MockupModule(nemo.core.NeuralModule):
        """
        Mockup component class.
        """

        def __init__(self):
            nemo.core.NeuralModule.__init__(self)

        def validate_params(self, params):
            """ Method for accessing private method of NeuralModuce class """
            return self._NeuralModule__validate_params(params)

    def setUp(self) -> None:
        super().setUp()

        # Mockup abstract methods.
        NeuralModuleConfigTest.MockupModule.__abstractmethods__ = set()

        # Create object.
        self.module = NeuralModuleConfigTest.MockupModule()

    def test_build_in_types(self):
        """ Tests whether build-in types are handled."""

        params = {"int": 123, "float": 12.4, "string": "ala ma kota", "bool": True}

        # Check error output.
        self.assertEqual(self.module.validate_params(params), True)

    def test_nested_dict(self):
        """ Tests whether (nested) dicts are handled."""

        params = {
            "dict_outer": {
                "dict_inner_1": {"int": 123, "float": 12.4, "string": "ala ma kota", "bool": True},
                "dict_inner_2": {"int": 123, "float": 12.4, "string": "ala ma kota", "bool": True},
            }
        }

        # Check error output.
        self.assertEqual(self.module.validate_params(params), True)

    def test_nested_list(self):
        """ Tests whether (nested) lists are handled."""

        params = {"list_outer": [[1, 2, 3, 4]]}

        # Check error output.
        self.assertEqual(self.module.validate_params(params), True)

    def test_nested_mix(self):
        """ Tests whether (nested) lists are handled."""

        params = {"list_outer": [{"int": 123, "float": 12.4, "string": "ala ma kota", "bool": True}]}

        # Check error output.
        self.assertEqual(self.module.validate_params(params), True)
