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

from ruamel import yaml

import nemo
from tests.common_setup import NeMoUnitTest


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

    def __extract_dict_from_handle_calls(self, handle):
        """ Helper method - extracts a ditionary from the handle containing several write calls. """
        # Put together the params - dict exported in several calls
        exported_dict = []
        for call in handle.write.mock_calls:
            call_value = str(call)[6:-2].replace(" ", "")
            # Drop '/n's.
            if call_value == "\\n":
                exported_dict.append(",")
                continue
            if call_value == "":
                continue
            if call_value in [",", ":", "{", "}", "[", "]"]:
                exported_dict.append(call_value)
            else:
                exported_dict.append("\"" + call_value + "\"")
        # "Preprocess" string.
        exported_string = "{" + ''.join(exported_dict)[:-1] + "}"
        exported_string = exported_string.replace(",,", ",")
        # print(exported_string)
        return eval(exported_string)

    @patch('__main__.__builtins__.open', new_callable=mock_open)
    def test_simple_export(self, mock_f):
        """
            Tests whether build-in types are properly exported.
            Mockup the hard disk write()/open() operations.
        """

        # params = {"int": 123, "float": 12.4, "string": "ala ma kota", "bool": True}
        module = NeuralModuleExportTest.MockupSimpleModule(123, 12.4, "ala_ma_kota", True)

        # Export.
        module.export_to_config("/tmp/simple_export.yml")

        # Assert that file that the file was "opened" in a write mode.
        mock_f.assert_called_with("/tmp/simple_export.yml", 'w')

        # Get handle to the call.
        handle = mock_f()

        # Get the exported dictionary.
        exported_dict = self.__extract_dict_from_handle_calls(handle)
        # print("exported_dict = ", exported_dict)

        # Assert that it contains main sections: header and init params.
        self.assertEqual("header" in exported_dict, True)
        self.assertEqual("init_params" in exported_dict, True)

        # Assert that the header contains class and spec.
        self.assertEqual("full_spec" in exported_dict["header"], True)

        # Check init params.
        exported_init_params = exported_dict["init_params"]
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
            exported_config = yaml.safe_load(stream)

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
            exported_config = yaml.safe_load(stream)

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
