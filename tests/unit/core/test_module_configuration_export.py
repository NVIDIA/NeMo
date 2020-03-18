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

import pytest
from ruamel.yaml import YAML

from nemo.core import NeuralModule

YAML = YAML(typ='safe')


@pytest.mark.usefixtures("neural_factory")
class TestNeuralModuleExport:
    """
        Class testing Neural Module configuration export.
    """

    class MockupSimpleModule(NeuralModule):
        """
        Mockup component class.
        """

        def __init__(self, a, b, c, d):
            super().__init__()

    def setup_method(self, method):
        """ 
            Setup_method is invoked for every test method of a class.
            Mocks up the module class.
        """
        # Mockup abstract methods.
        TestNeuralModuleExport.MockupSimpleModule.__abstractmethods__ = set()

    @pytest.mark.unit
    def test_simple_export(self, tmpdir):
        """
            Tests whether build-in types are properly exported.

            Args:
                tmpdir: Fixture which will provide a temporary directory.
        """

        # Set params: {"int": 123, "float": 12.4, "string": "ala ma kota", "bool": True}
        params = {"a": 123, "b": 12.4, "c": "ala ma kota", "d": True}
        module = TestNeuralModuleExport.MockupSimpleModule(**params)

        # Generate filename in the temporary directory.
        tmp_file_name = str(tmpdir.mkdir("export").join("simple_export.yml"))
        # Export.
        module.export_to_config(tmp_file_name)

        # Check the resulting config file.
        with open(tmp_file_name, 'r') as stream:
            exported_config = YAML.load(stream)

        # Assert that it contains main sections: header and init params.
        assert "header" in exported_config
        assert "init_params" in exported_config

        # Assert that the header contains class and spec.
        assert "full_spec" in exported_config["header"]

        # Check init params.
        exported_init_params = exported_config["init_params"]
        assert int(exported_init_params["a"]) == 123
        assert float(exported_init_params["b"]) == 12.4
        assert exported_init_params["c"] == "ala ma kota"
        assert bool(exported_init_params["d"]) == True

    @pytest.mark.unit
    def test_nested_list_export(self, tmpdir):
        """
            Tests whether (nested*) lists are properly exported.

            Args:
                tmpdir: Fixture which will provide a temporary directory.
        """

        # Params: list, list of lists, list of lists of lists, None type!
        module = TestNeuralModuleExport.MockupSimpleModule(
            a=[123], b=[[12.4]], c=[[["ala", "ma", "kota"], "kot ma"], "ale"], d=None
        )

        # Generate filename in the temporary directory.
        tmp_file_name = str(tmpdir.mkdir("export").join("nested_list_export.yml"))
        # Export.
        module.export_to_config(tmp_file_name)

        # Check the resulting config file.
        with open(tmp_file_name, 'r') as stream:
            exported_config = YAML.load(stream)

        # Assert that it contains main sections: header and init params.
        assert "header" in exported_config
        assert "init_params" in exported_config

        # Check init params.
        exported_init_params = exported_config["init_params"]
        assert exported_init_params["a"][0] == 123
        assert exported_init_params["b"][0][0] == 12.4
        assert exported_init_params["c"][0][0][0] == "ala"
        assert exported_init_params["c"][0][0][1] == "ma"
        assert exported_init_params["c"][0][0][2] == "kota"
        assert exported_init_params["c"][0][1] == "kot ma"
        assert exported_init_params["c"][1] == "ale"
        assert exported_init_params["d"] == None

    @pytest.mark.unit
    def test_nested_dict_export(self, tmpdir):
        """
            Tests whether (nested*) dictionaries are properly exported.

            Args:
                tmpdir: Fixture which will provide a temporary directory.
        """

        # Params: dict, dict with list, dict with dict, build-in.
        module = TestNeuralModuleExport.MockupSimpleModule(
            a={"int": 123}, b={"floats": [12.4, 71.2]}, c={"ala": {"ma": "kota", "nie_ma": "psa"}}, d=True
        )

        # Generate filename in the temporary directory.
        tmp_file_name = str(tmpdir.mkdir("export").join("nested_dict_export.yml"))
        # Export.
        module.export_to_config(tmp_file_name)

        # Check the resulting config file.
        with open(tmp_file_name, 'r') as stream:
            exported_config = YAML.load(stream)

        # Assert that it contains main sections: header and init params.
        assert "header" in exported_config
        assert "init_params" in exported_config

        # Check init params.
        exported_init_params = exported_config["init_params"]
        assert exported_init_params["a"]["int"] == 123
        assert exported_init_params["b"]["floats"][0] == 12.4
        assert exported_init_params["b"]["floats"][1] == 71.2
        assert exported_init_params["c"]["ala"]["ma"] == "kota"
        assert exported_init_params["c"]["ala"]["nie_ma"] == "psa"
        assert exported_init_params["d"]

    @pytest.mark.unit
    def test_unallowed_export(self, tmpdir):
        """
            Tests whether unallowed types are NOT exported.

            Args:
                tmpdir: Fixture which will provide a temporary directory.
        """

        e = Exception("some random object")

        # Params: dict, dict with list, dict with dict, build-in.
        module = TestNeuralModuleExport.MockupSimpleModule(e, False, False, False)

        # Generate filename in the temporary directory.
        tmp_file_name = str(tmpdir.mkdir("export").join("unallowed_export.yml"))
        # Assert export error.
        with pytest.raises(ValueError):
            module.export_to_config(tmp_file_name)
