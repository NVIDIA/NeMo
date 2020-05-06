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

from nemo.core import NeuralModule


@pytest.mark.usefixtures("neural_factory")
class TestNeuralModuleImport:
    """
        Class testing Neural Module configuration export.
    """

    class FirstSimpleModule(NeuralModule):
        """
        Mockup component class.
        """

        def __init__(self, a, b, c, d):
            super().__init__()

    class SecondSimpleModule(NeuralModule):
        """
        Mockup component class.
        """

        def __init__(self, x, y):
            super().__init__()

    def setup_method(self, method):
        """ 
            Setup_method is invoked for every test method of a class.
            Mocks up the classes.
        """
        # Mockup abstract methods.
        TestNeuralModuleImport.FirstSimpleModule.__abstractmethods__ = set()
        TestNeuralModuleImport.SecondSimpleModule.__abstractmethods__ = set()

    @pytest.mark.unit
    def test_simple_import_root_neural_module(self, tmpdir):
        """
            Tests whether the Neural Module can instantiate a simple module by loading a configuration file.

            Args:
                tmpdir: Fixture which will provide a temporary directory.
        """

        # params = {"int": 123, "float": 12.4, "string": "ala ma kota", "bool": True}
        orig_module = TestNeuralModuleImport.FirstSimpleModule(123, 12.4, "ala ma kota", True)

        # Generate filename in the temporary directory.
        tmp_file_name = str(tmpdir.mkdir("export").join("simple_import_root.yml"))
        # Export.
        orig_module.export_to_config(tmp_file_name)

        # Import and create the new object.
        new_module = NeuralModule.import_from_config(tmp_file_name)

        # Compare class types.
        assert type(orig_module).__name__ == type(new_module).__name__

        # Compare objects - by its all params.
        param_keys = orig_module.init_params.keys()
        for key in param_keys:
            assert orig_module.init_params[key] == new_module.init_params[key]

    @pytest.mark.unit
    def test_simple_import_leaf_module(self, tmpdir):
        """
            Tests whether a particular module can instantiate another
            instance (a copy) by loading a configuration file.

            Args:
                tmpdir: Fixture which will provide a temporary directory.
        """

        # params = {"int": 123, "float": 12.4, "string": "ala ma kota", "bool": True}
        orig_module = TestNeuralModuleImport.FirstSimpleModule(123, 12.4, "ala ma kota", True)

        # Generate filename in the temporary directory.
        tmp_file_name = str(tmpdir.mkdir("export").join("simple_import_leaf.yml"))
        # Export.
        orig_module.export_to_config(tmp_file_name)

        # Import and create the new object.
        new_module = TestNeuralModuleImport.FirstSimpleModule.import_from_config(tmp_file_name)

        # Compare class types.
        assert type(orig_module).__name__ == type(new_module).__name__

        # Compare objects - by its all params.
        param_keys = orig_module.init_params.keys()
        for key in param_keys:
            assert orig_module.init_params[key] == new_module.init_params[key]

    @pytest.mark.unit
    def test_incompatible_import_leaf_module(self, tmpdir):
        """
            Tests whether a particular module can instantiate another
            instance (a copy) by loading a configuration file.

            Args:
                tmpdir: Fixture which will provide a temporary directory.
        """

        # params = {"int": 123, "float": 12.4, "string": "ala ma kota", "bool": True}
        orig_module = TestNeuralModuleImport.SecondSimpleModule(["No", "way", "dude!"], None)

        # Generate filename in the temporary directory.
        tmp_file_name = str(tmpdir.mkdir("export").join("incompatible_import_leaf.yml"))
        # Export.
        orig_module.export_to_config(tmp_file_name)

        # This will actuall create an instance of SecondSimpleModule - OK.
        new_module = NeuralModule.import_from_config(tmp_file_name)
        # Compare class types.
        assert type(orig_module).__name__ == type(new_module).__name__

        # This will create an instance of SecondSimpleModule, not FirstSimpleModule - SO NOT OK!!
        with pytest.raises(ImportError):
            _ = TestNeuralModuleImport.FirstSimpleModule.import_from_config(tmp_file_name)
