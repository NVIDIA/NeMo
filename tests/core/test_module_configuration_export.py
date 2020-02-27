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

from unittest.mock import mock_open, patch

from ruamel.yaml import YAML

import nemo
from tests.common_setup import NeMoUnitTest

YAML = YAML(typ='safe')


class NeuralModuleExportTest(NeMoUnitTest):
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
        NeuralModuleExportTest.MockupSimpleModule.__abstractmethods__ = set()

    def test_simple_export(self):
        """
            Tests whether build-in types are properly exported.
        """

        # params = {"int": 123, "float": 12.4, "string": "ala ma kota", "bool": True}
        module = NeuralModuleExportTest.MockupSimpleModule(123, 12.4, "ala_ma_kota", True)

        # Export.
        module.export_to_config("/tmp/simple_export.yml")

        # Check the resulting config file.
        with open("/tmp/simple_export.yml", 'r') as stream:
            exported_config = YAML.load(stream)

        # Assert that it contains main sections: header and init params.
        self.assertEqual("header" in exported_config, True)
        self.assertEqual("init_params" in exported_config, True)

        # Assert that the header contains class and spec.
        self.assertEqual("full_spec" in exported_config["header"], True)

        # Check init params.
        exported_init_params = exported_config["init_params"]
        self.assertEqual(int(exported_init_params["a"]), 123)
        self.assertEqual(float(exported_init_params["b"]), 12.4)
        self.assertEqual(exported_init_params["c"], "ala_ma_kota")
        self.assertEqual(bool(exported_init_params["d"]), True)

    def test_nested_list_export(self):
        """ Tests whether (nested*) lists are properly exported."""

        # Params: list, list of lists, list of lists of lists, None type!
        module = NeuralModuleExportTest.MockupSimpleModule(
            a=[123], b=[[12.4]], c=[[["ala", "ma", "kota"], "kot ma"], "ale"], d=None
        )

        # Export.
        module.export_to_config("/tmp/nested_list_export.yml")

        # Check the resulting config file.
        with open("/tmp/nested_list_export.yml", 'r') as stream:
            exported_config = YAML.load(stream)

        # Assert that it contains main sections: header and init params.
        self.assertEqual("header" in exported_config, True)
        self.assertEqual("init_params" in exported_config, True)

        # Check init params.
        exported_init_params = exported_config["init_params"]
        self.assertEqual(exported_init_params["a"][0], 123)
        self.assertEqual(exported_init_params["b"][0][0], 12.4)
        self.assertEqual(exported_init_params["c"][0][0][0], "ala")
        self.assertEqual(exported_init_params["c"][0][0][1], "ma")
        self.assertEqual(exported_init_params["c"][0][0][2], "kota")
        self.assertEqual(exported_init_params["c"][0][1], "kot ma")
        self.assertEqual(exported_init_params["c"][1], "ale")
        self.assertEqual(exported_init_params["d"], None)

    def test_nested_dict_export(self):
        """ Tests whether (nested*) dictionaries are properly exported."""

        # Params: dict, dict with list, dict with dict, build-in.
        module = NeuralModuleExportTest.MockupSimpleModule(
            a={"int": 123}, b={"floats": [12.4, 71.2]}, c={"ala": {"ma": "kota", "nie_ma": "psa"}}, d=True
        )

        # Export.
        module.export_to_config("/tmp/nested_dict_export.yml")

        # Check the resulting config file.
        with open("/tmp/nested_dict_export.yml", 'r') as stream:
            exported_config = YAML.load(stream)

        # Assert that it contains main sections: header and init params.
        self.assertEqual("header" in exported_config, True)
        self.assertEqual("init_params" in exported_config, True)

        # Check init params.
        exported_init_params = exported_config["init_params"]
        self.assertEqual(exported_init_params["a"]["int"], 123)
        self.assertEqual(exported_init_params["b"]["floats"][0], 12.4)
        self.assertEqual(exported_init_params["b"]["floats"][1], 71.2)
        self.assertEqual(exported_init_params["c"]["ala"]["ma"], "kota")
        self.assertEqual(exported_init_params["c"]["ala"]["nie_ma"], "psa")
        self.assertEqual(exported_init_params["d"], True)

    def test_unallowed_export(self):
        """ Tests whether unallowed types are NOT exported."""

        e = Exception("some random object")

        # Params: dict, dict with list, dict with dict, build-in.
        module = NeuralModuleExportTest.MockupSimpleModule(e, False, False, False)

        # Assert export error.
        with self.assertRaises(ValueError):
            module.export_to_config("/tmp/unallowed_export.yml")
