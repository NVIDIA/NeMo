# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import tempfile
from dataclasses import dataclass
from dataclasses import dataclass as python_dataclass
from typing import Any, Dict, List, Optional

import pytest

from nemo.tron.utils.config_utils import ConfigContainer
from nemo.tron.utils.instantiate_utils import InstantiationMode


# Test fixtures - example config classes
@dataclass
class SimpleConfig(ConfigContainer):
    """Simple test config class with basic fields."""

    name: str
    value: int
    enabled: bool = True


@dataclass
class ConfigWithValidation(ConfigContainer):
    """Config class with custom validation."""

    min_value: int
    max_value: int

    def validation(self):
        if self.min_value > self.max_value:
            raise ValueError("min_value cannot be greater than max_value")
        if self.min_value < 0:
            raise ValueError("min_value cannot be negative")


# Regular Python dataclass (not a ConfigContainer)
@python_dataclass
class RegularDataclass:
    """A regular Python dataclass, not a ConfigContainer."""

    name: str
    value: int

    def update_name(self, new_name):
        """Update the name field."""
        self.name = new_name


# Nested regular dataclass
@python_dataclass
class NestedRegularDataclass:
    """A regular dataclass with a nested regular dataclass."""

    title: str
    inner: RegularDataclass


@dataclass
class ConfigWithRegularDataclass(ConfigContainer):
    """Config class that contains a regular dataclass."""

    name: str
    regular_data: RegularDataclass


@dataclass
class ConfigWithNestedRegularDataclass(ConfigContainer):
    """Config class with nested regular dataclasses."""

    name: str
    nested_data: NestedRegularDataclass


@dataclass
class NestedConfig(ConfigContainer):
    """Config class with nested ConfigContainer."""

    name: str
    simple_config: SimpleConfig
    configs: List[SimpleConfig] = None


@dataclass
class ComplexConfig(ConfigContainer):
    """More complex config with various field types and nested configs."""

    name: str
    simple_config: SimpleConfig
    nested_config: Optional[NestedConfig] = None
    values: List[int] = None
    mapping: Dict[str, Any] = None


# Pytest fixtures to replace setUp method
@pytest.fixture
def simple_config():
    """Return a simple config instance."""
    return SimpleConfig(name="test", value=42)


@pytest.fixture
def valid_config():
    """Return a valid config with validation."""
    return ConfigWithValidation(min_value=1, max_value=10)


@pytest.fixture
def nested_config():
    """Return a nested config instance."""
    return NestedConfig(name="nested", simple_config=SimpleConfig(name="inner", value=21))


@pytest.fixture
def complex_config(simple_config, nested_config):
    """Return a complex config instance."""
    return ComplexConfig(
        name="complex",
        simple_config=simple_config,
        nested_config=nested_config,
        values=[1, 2, 3],
        mapping={"key1": "value1", "key2": 2},
    )


@pytest.fixture
def regular_dataclass():
    """Return a regular dataclass instance."""
    return RegularDataclass(name="regular", value=100)


@pytest.fixture
def nested_regular_dataclass(regular_dataclass):
    """Return a nested regular dataclass."""
    return NestedRegularDataclass(title="parent", inner=RegularDataclass(name="child", value=50))


@pytest.fixture
def config_with_regular_dataclass(regular_dataclass):
    """Return a config containing a regular dataclass."""
    return ConfigWithRegularDataclass(name="container", regular_data=regular_dataclass)


@pytest.fixture
def config_with_nested_regular(nested_regular_dataclass):
    """Return a config containing a nested regular dataclass."""
    return ConfigWithNestedRegularDataclass(name="container", nested_data=nested_regular_dataclass)


def test_basic_initialization():
    """Test basic initialization of config classes."""
    config = SimpleConfig(name="test", value=42)
    assert config.name == "test"
    assert config.value == 42
    assert config.enabled is True  # Default value


def test_nested_config():
    """Test nested config containers."""
    config = NestedConfig(name="parent", simple_config=SimpleConfig(name="child", value=10))
    assert config.name == "parent"
    assert config.simple_config.name == "child"
    assert config.simple_config.value == 10


def test_to_dict_simple():
    """Test conversion to dictionary for simple config."""
    config = SimpleConfig(name="test", value=42)
    config_dict = config.to_dict()

    assert config_dict["name"] == "test"
    assert config_dict["value"] == 42
    assert config_dict["enabled"] is True
    assert config_dict["_target_"] == "utils.test_config_utils.SimpleConfig"


def test_to_dict_nested(complex_config):
    """Test conversion to dictionary with nested configs."""
    # Convert to dict
    config_dict = complex_config.to_dict()

    # Check top level
    assert config_dict["name"] == "complex"

    # Check nested simple_config
    assert config_dict["simple_config"]["name"] == "test"
    assert config_dict["simple_config"]["value"] == 42

    # Check nested_config and its nested simple_config
    assert config_dict["nested_config"]["name"] == "nested"
    assert config_dict["nested_config"]["simple_config"]["name"] == "inner"
    assert config_dict["nested_config"]["simple_config"]["value"] == 21

    # Check lists and dicts
    assert config_dict["values"] == [1, 2, 3]
    assert config_dict["mapping"] == {"key1": "value1", "key2": 2}


def test_from_dict_simple():
    """Test creating config from dictionary for simple config."""
    config_dict = {
        "_target_": "tests.tron.utils.test_config_utils.SimpleConfig",
        "name": "from_dict",
        "value": 100,
        "enabled": False,
    }

    config = SimpleConfig.from_dict(config_dict)
    assert config.name == "from_dict"
    assert config.value == 100
    assert config.enabled is False


def test_from_dict_with_target():
    """Test from_dict with _target_ field."""
    config_dict = {
        "_target_": "tests.tron.utils.test_config_utils.SimpleConfig",
        "name": "from_dict",
        "value": 100,
        "enabled": False,
    }

    config = SimpleConfig.from_dict(config_dict)
    assert config.name == "from_dict"
    assert config.value == 100
    assert config.enabled is False


def test_from_dict_nested():
    """Test from_dict with nested configs."""
    config_dict = {
        "_target_": "tests.tron.utils.test_config_utils.NestedConfig",
        "name": "parent",
        "simple_config": {
            "_target_": "tests.tron.utils.test_config_utils.SimpleConfig",
            "name": "child",
            "value": 200,
            "enabled": True,
        },
    }

    config = NestedConfig.from_dict(config_dict)
    assert config.name == "parent"
    assert config.simple_config.name == "child"
    assert config.simple_config.value == 200
    assert config.simple_config.enabled is True


def test_from_dict_strict_mode():
    """Test strict mode for from_dict."""
    # Extra field in dict
    config_dict = {
        "_target_": "tests.tron.utils.test_config_utils.SimpleConfig",
        "name": "test",
        "value": 42,
        "extra_field": "extra",  # Not in SimpleConfig
    }

    # Should raise ValueError in strict mode
    with pytest.raises(ValueError, match="Dictionary contains extra keys"):
        SimpleConfig.from_dict(config_dict, mode=InstantiationMode.STRICT)

    # Should succeed in lenient mode
    config = SimpleConfig.from_dict(config_dict, mode=InstantiationMode.LENIENT)
    assert config.name == "test"
    assert config.value == 42
    # extra_field is ignored


def test_yaml_roundtrip(complex_config):
    """Test saving and loading from YAML."""
    # Create temp file for YAML
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp:
        tmp_path = tmp.name

    try:
        # Save to YAML
        complex_config.to_yaml(tmp_path)

        # Verify file exists and has content
        assert os.path.exists(tmp_path)
        with open(tmp_path, "r") as f:
            yaml_content = f.read()
            assert "name: complex" in yaml_content

        # Load from YAML
        loaded_config = ComplexConfig.from_yaml(tmp_path)

        # Check loaded config
        assert loaded_config.name == "complex"
        assert loaded_config.simple_config.name == "test"
        assert loaded_config.simple_config.value == 42
        assert loaded_config.nested_config.name == "nested"
        assert loaded_config.nested_config.simple_config.name == "inner"
        assert loaded_config.nested_config.simple_config.value == 21
        assert loaded_config.values == [1, 2, 3]
        assert loaded_config.mapping == {"key1": "value1", "key2": 2}
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_deepcopy(complex_config):
    """Test deep copying of configs."""
    import copy

    original = complex_config
    copied = copy.deepcopy(original)

    # Values should be the same initially
    assert copied.name == original.name
    assert copied.simple_config.name == original.simple_config.name

    # Modify the copy
    copied.name = "modified"
    copied.simple_config.name = "modified_inner"

    # Original should remain unchanged
    assert original.name == "complex"
    assert original.simple_config.name == "test"

    # Copy should have new values
    assert copied.name == "modified"
    assert copied.simple_config.name == "modified_inner"


def test_from_yaml_nonexistent_file():
    """Test from_yaml with nonexistent file."""
    with pytest.raises(FileNotFoundError):
        SimpleConfig.from_yaml("nonexistent_file.yaml")


def test_to_dict_with_regular_dataclass():
    """Test that to_dict properly converts regular dataclasses to dictionaries."""
    # Create a regular dataclass
    regular = RegularDataclass(name="test", value=42)

    # Create a config with the regular dataclass
    config = ConfigWithRegularDataclass(name="container", regular_data=regular)

    # Convert to dictionary
    config_dict = config.to_dict()

    # Check that the regular dataclass was converted to a dictionary
    assert isinstance(config_dict["regular_data"], dict)
    assert config_dict["regular_data"]["name"] == "test"
    assert config_dict["regular_data"]["value"] == 42

    # Test with nested regular dataclass
    inner = RegularDataclass(name="inner", value=10)
    nested = NestedRegularDataclass(title="outer", inner=inner)
    config = ConfigWithNestedRegularDataclass(name="nested-container", nested_data=nested)

    # Convert to dictionary
    config_dict = config.to_dict()

    # Check that the nested structure was properly converted
    assert isinstance(config_dict["nested_data"], dict)
    assert config_dict["nested_data"]["title"] == "outer"
    assert isinstance(config_dict["nested_data"]["inner"], dict)
    assert config_dict["nested_data"]["inner"]["name"] == "inner"
    assert config_dict["nested_data"]["inner"]["value"] == 10


def test_roundtrip_with_regular_dataclass():
    """Test round-trip conversion (to_dict -> from_dict) with regular dataclasses."""
    # Create original config with regular dataclass
    inner = RegularDataclass(name="inner", value=10)
    nested = NestedRegularDataclass(title="outer", inner=inner)
    config = ConfigWithNestedRegularDataclass(name="nested-container", nested_data=nested)

    # Convert to dictionary and back
    config_dict = config.to_dict()

    # Verify _target_ fields were automatically added
    assert config_dict["_target_"] == "utils.test_config_utils.ConfigWithNestedRegularDataclass"
    assert config_dict["nested_data"]["_target_"] == "utils.test_config_utils.NestedRegularDataclass"
    assert config_dict["nested_data"]["inner"]["_target_"] == "utils.test_config_utils.RegularDataclass"

    # Now convert back
    round_trip = ConfigWithNestedRegularDataclass.from_dict(config_dict)

    # Verify the structure is preserved
    assert round_trip.name == "nested-container"
    assert isinstance(round_trip.nested_data, NestedRegularDataclass)
    assert round_trip.nested_data.title == "outer"
    assert isinstance(round_trip.nested_data.inner, RegularDataclass)
    assert round_trip.nested_data.inner.name == "inner"
    assert round_trip.nested_data.inner.value == 10


def test_integration_with_instantiate():
    """Test integration with instantiate module."""
    # Create a complex config with nested structures
    config_dict = {
        "_target_": "tests.tron.utils.test_config_utils.ComplexConfig",
        "name": "instantiated",
        "simple_config": {
            "_target_": "utils.test_config_utils.SimpleConfig",
            "name": "child",
            "value": 100,
        },
        "nested_config": {
            "_target_": "utils.test_config_utils.NestedConfig",
            "name": "nested",
            "simple_config": {
                "_target_": "utils.test_config_utils.SimpleConfig",
                "name": "inner",
                "value": 42,
            },
        },
        "values": [1, 2, 3, 4],
    }

    # Use from_dict which now uses instantiate
    config = ComplexConfig.from_dict(config_dict)

    # Verify all fields were properly instantiated
    assert config.name == "instantiated"
    assert isinstance(config.simple_config, SimpleConfig)
    assert config.simple_config.name == "child"
    assert config.simple_config.value == 100

    assert isinstance(config.nested_config, NestedConfig)
    assert config.nested_config.name == "nested"
    assert isinstance(config.nested_config.simple_config, SimpleConfig)
    assert config.nested_config.simple_config.name == "inner"
    assert config.nested_config.simple_config.value == 42

    assert config.values == [1, 2, 3, 4]

    # Verify round-trip to dict and back works
    dict_form = config.to_dict()
    round_trip = ComplexConfig.from_dict(dict_form)

    assert round_trip.name == config.name
    assert round_trip.simple_config.name == config.simple_config.name
    assert round_trip.nested_config.simple_config.value == config.nested_config.simple_config.value


def test_complex_nested_dataclass_conversion():
    """Test conversion of more complex nested dataclass structures."""
    # Create a more complex nested structure with multiple levels
    class_a = RegularDataclass(name="a", value=1)
    class_b = RegularDataclass(name="b", value=2)
    class_c = RegularDataclass(name="c", value=3)

    # Create a list of dataclasses
    dataclass_list = [class_a, class_b]

    # Create a dict with dataclasses as values
    dataclass_dict = {"first": class_a, "second": class_b}

    # Create a nested structure with these collections
    @python_dataclass
    class ComplexDataclass:
        name: str
        items: List[RegularDataclass]
        mapping: Dict[str, RegularDataclass]
        nested: RegularDataclass

    complex_instance = ComplexDataclass(name="complex", items=dataclass_list, mapping=dataclass_dict, nested=class_c)

    # Create a config with this complex instance
    @dataclass
    class ConfigWithComplexDataclass(ConfigContainer):
        name: str
        complex_data: ComplexDataclass

    config = ConfigWithComplexDataclass(name="container", complex_data=complex_instance)

    # Convert to dictionary
    config_dict = config.to_dict()

    # Verify the complex structure was properly converted
    assert config_dict["name"] == "container"
    assert config_dict["complex_data"]["name"] == "complex"

    # Check list conversion
    assert isinstance(config_dict["complex_data"]["items"], list)
    assert len(config_dict["complex_data"]["items"]) == 2
    assert config_dict["complex_data"]["items"][0]["name"] == "a"
    assert config_dict["complex_data"]["items"][1]["name"] == "b"

    # Check dict conversion
    assert isinstance(config_dict["complex_data"]["mapping"], dict)
    assert config_dict["complex_data"]["mapping"]["first"]["name"] == "a"
    assert config_dict["complex_data"]["mapping"]["second"]["name"] == "b"

    # Check nested dataclass
    assert config_dict["complex_data"]["nested"]["name"] == "c"
    assert config_dict["complex_data"]["nested"]["value"] == 3


def test_from_dict_with_default_field():
    """Test from_dict with fields that use default values."""
    # Create a config with only required fields
    config_dict = {
        "_target_": "tests.tron.utils.test_config_utils.SimpleConfig",
        "name": "minimal",
        "value": 50,
        # 'enabled' field is missing but has a default
    }

    # Should use the default value for 'enabled'
    config = SimpleConfig.from_dict(config_dict)
    assert config.name == "minimal"
    assert config.value == 50
    assert config.enabled is True  # Default value should be used


def test_correct_target_check():
    """Test handling of correct _target_ field."""
    # Create a config dictionary with proper target
    config_dict = {
        "_target_": "tests.tron.utils.test_config_utils.SimpleConfig",
        "name": "test",
        "value": 42,
    }

    # Should create the instance without any issues
    config = SimpleConfig.from_dict(config_dict)
    assert config.name == "test"
    assert config.value == 42


# Class with underscore fields for testing lines 165-166
@python_dataclass
class DataclassWithUnderscoreFields:
    """A regular dataclass with fields that start with underscore."""

    name: str  # Regular field
    value: int  # Regular field
    _private: str  # Field starting with underscore
    __dunder__: int  # Double underscore field

    def method(self):
        """Regular method."""
        return self.name


@dataclass
class ConfigWithUnderscoreDataclass(ConfigContainer):
    """Config container with a dataclass that has underscore fields."""

    title: str
    data: DataclassWithUnderscoreFields


def test_underscore_fields_skipped():
    """Test that fields starting with underscore are skipped when converting to dict (lines 165-166)."""
    # Create test dataclass with underscore fields
    test_data = DataclassWithUnderscoreFields(name="test_name", value=42, _private="should_be_skipped", __dunder__=100)

    # Create a config container with the test dataclass
    config = ConfigWithUnderscoreDataclass(title="Test Config", data=test_data)

    # Convert to dictionary
    result = config.to_dict()

    # Check that the result has the expected structure
    assert "_target_" in result
    assert "title" in result
    assert "data" in result
    assert "_target_" in result["data"]
    assert "name" in result["data"]
    assert "value" in result["data"]

    # Critical test for lines 165-166: verify underscore fields are skipped
    assert "_private" not in result["data"]
    assert "__dunder__" not in result["data"]

    # Direct test of the _convert_value_to_dict method
    converted = ConfigContainer._convert_value_to_dict(test_data)
    assert "name" in converted
    assert "value" in converted
    assert "_private" not in converted
    assert "__dunder__" not in converted


def test_convert_value_to_dict_skips_underscore_fields():
    """Directly test the _convert_value_to_dict method's skipping of underscore fields (lines 165-166)."""
    # Create a more complex dataclass with underscore fields at different nesting levels

    @python_dataclass
    class InnerDataclass:
        regular_field: str
        _private_field: str

    @python_dataclass
    class NestedWithUnderscores:
        name: str
        inner: InnerDataclass
        _skipped: str
        regular_list: List[str]
        _private_list: List[str]
        mixed_dict: Dict[str, Any]

    # Create test instance
    inner = InnerDataclass(regular_field="visible", _private_field="invisible")
    test_obj = NestedWithUnderscores(
        name="test",
        inner=inner,
        _skipped="should not appear",
        regular_list=["a", "b"],
        _private_list=["hidden1", "hidden2"],
        mixed_dict={
            "normal": "shown",
            "_private": "still shown because it's a dict key, not a field",
            "nested": InnerDataclass(regular_field="nested visible", _private_field="nested invisible"),
        },
    )

    # Call the method directly
    result = ConfigContainer._convert_value_to_dict(test_obj)

    # Verify fields starting with underscore are skipped at all levels
    assert "_target_" in result
    assert "name" in result
    assert "inner" in result
    assert "_skipped" not in result
    assert "regular_list" in result
    assert "_private_list" not in result
    assert "mixed_dict" in result

    # Check nested dataclass
    assert "_target_" in result["inner"]
    assert "regular_field" in result["inner"]
    assert "_private_field" not in result["inner"]

    # Check dictionary values - underscore fields in nested dataclasses should be skipped
    nested_in_dict = result["mixed_dict"]["nested"]
    assert "_target_" in nested_in_dict
    assert "regular_field" in nested_in_dict
    assert "_private_field" not in nested_in_dict

    # Dictionary keys with underscores should be preserved (not affected by lines 165-166)
    assert "_private" in result["mixed_dict"]
