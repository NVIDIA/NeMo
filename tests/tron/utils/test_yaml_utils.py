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

import inspect
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from nemo.tron.utils.yaml_utils import (
    _function_representer,
    _safe_object_representer,
    _torch_dtype_representer,
    safe_yaml_representers,
)


# Test fixtures
class DummyClass:
    def test_method(self):
        pass

    @staticmethod
    def static_method():
        pass

    @classmethod
    def class_method(cls):
        pass


@dataclass
class DummyDataClass:
    name: str
    value: int


def dummy_function():
    pass


def dummy_function_with_args(a, b=2, c=3):
    return a + b + c


def test_context_manager_preserves_representers():
    """Test that the context manager preserves and restores original representers."""
    # Store original representers count
    original_rep_count = len(yaml.SafeDumper.yaml_representers)
    original_multi_rep_count = len(yaml.SafeDumper.yaml_multi_representers)

    # Use context manager
    with safe_yaml_representers():
        # Should have added new representers
        assert len(yaml.SafeDumper.yaml_representers) > original_rep_count
        assert len(yaml.SafeDumper.yaml_multi_representers) > original_multi_rep_count

    # After context, should be back to original counts
    assert len(yaml.SafeDumper.yaml_representers) == original_rep_count
    assert len(yaml.SafeDumper.yaml_multi_representers) == original_multi_rep_count


def test_function_yaml_dump():
    """Test the function representer using yaml.safe_dump."""
    with safe_yaml_representers():
        result = yaml.safe_dump(dummy_function)
        assert "_target_" in result
        assert "test_yaml_utils.dummy_function" in result
        assert "_call_: false" in result


def test_class_representer():
    """Test the class representer."""
    with safe_yaml_representers():
        result = yaml.safe_dump(DummyClass)
        assert "_target_" in result
        assert "test_yaml_utils.DummyClass" in result
        assert "_call_: false" in result


def test_instance_representer():
    """Test the instance representer."""
    instance = DummyClass()
    with safe_yaml_representers():
        result = yaml.safe_dump(instance)
        assert "_target_" in result
        assert "test_yaml_utils.DummyClass" in result
        assert "_call_: true" in result


def test_dataclass_representer():
    """Test representation of dataclasses."""
    instance = DummyDataClass(name="test", value=42)
    with safe_yaml_representers():
        result = yaml.safe_dump(instance)
        assert "_target_" in result
        assert "test_yaml_utils.DummyDataClass" in result
        assert "_call_: true" in result


def test_nested_objects():
    """Test representation of nested objects."""
    data = {
        "function": dummy_function,
        "class": DummyClass,
        "instance": DummyClass(),
        "dataclass": DummyDataClass(name="nested", value=100),
    }

    with safe_yaml_representers():
        result = yaml.safe_dump(data)
        assert "function" in result
        assert "class" in result
        assert "instance" in result
        assert "dataclass" in result
        assert "test_yaml_utils.dummy_function" in result
        assert "test_yaml_utils.DummyClass" in result
        assert "test_yaml_utils.DummyDataClass" in result


def test_safe_yaml_representers_context():
    """Test that the context manager properly adds and removes representers."""
    # Check that custom representers are not registered initially
    assert type(lambda: ...) not in yaml.SafeDumper.yaml_representers

    # Enter context manager
    with safe_yaml_representers():
        # Check that custom representers are registered
        assert type(lambda: ...) in yaml.SafeDumper.yaml_representers

        # Test dumping a function
        result = yaml.safe_dump(dummy_function)
        assert "_target_:" in result
        assert "test_yaml_utils.dummy_function" in result
        assert "_call_: false" in result

    # Check that custom representers are removed after context
    assert type(lambda: ...) not in yaml.SafeDumper.yaml_representers


def test_function_representer():
    """Test the function representer."""
    # Create a dummy dumper
    dumper = yaml.SafeDumper(None)

    # Test with a regular function
    result = _function_representer(dumper, dummy_function)

    # Verify the result - check that it's a MappingNode
    assert hasattr(result, "value")
    assert len(result.value) > 0

    # Extract the values from the node
    mapping_data = {}
    for key_node, value_node in result.value:
        mapping_data[key_node.value] = value_node.value

    # Verify the specific values
    expected_target = f"{inspect.getmodule(dummy_function).__name__}.{dummy_function.__qualname__}"
    assert mapping_data["_target_"] == expected_target
    # YAML ScalarNode for booleans might be 'false' as string, not False as Python bool
    assert mapping_data["_call_"] == "false" or mapping_data["_call_"] is False

    # Test with a lambda function
    lambda_func = lambda x: x + 1  # noqa: E731
    result = _function_representer(dumper, lambda_func)

    # Extract the values from the node
    mapping_data = {}
    for key_node, value_node in result.value:
        mapping_data[key_node.value] = value_node.value

    # Verify the lambda function
    assert mapping_data["_target_"].endswith(lambda_func.__qualname__)
    assert mapping_data["_call_"] == "false" or mapping_data["_call_"] is False

    # Test with a method
    method = DummyClass.test_method
    result = _function_representer(dumper, method)

    # Extract the values from the node
    mapping_data = {}
    for key_node, value_node in result.value:
        mapping_data[key_node.value] = value_node.value

    # Verify the method
    expected_target = f"{inspect.getmodule(method).__name__}.{method.__qualname__}"
    assert mapping_data["_target_"] == expected_target
    assert mapping_data["_call_"] == "false" or mapping_data["_call_"] is False


def test_torch_dtype_representer():
    """Test the torch dtype representer."""
    try:
        import torch

        # Create a dummy dumper
        dumper = yaml.SafeDumper(None)

        # Test with torch.float32
        dtype = torch.float32
        result = _torch_dtype_representer(dumper, dtype)

        # Verify the result - check that it's a MappingNode
        assert hasattr(result, "value")
        assert len(result.value) > 0

        # Extract the values from the node
        mapping_data = {}
        for key_node, value_node in result.value:
            mapping_data[key_node.value] = value_node.value

        # Verify the result
        assert mapping_data["_target_"] == str(dtype)
        # YAML ScalarNode for booleans might be 'false' as string, not False as Python bool
        assert mapping_data["_call_"] == "false" or mapping_data["_call_"] is False

        # Test with torch.int64
        dtype = torch.int64
        result = _torch_dtype_representer(dumper, dtype)

        # Extract the values from the node
        mapping_data = {}
        for key_node, value_node in result.value:
            mapping_data[key_node.value] = value_node.value

        # Verify the result
        assert mapping_data["_target_"] == str(dtype)
        assert mapping_data["_call_"] == "false" or mapping_data["_call_"] is False

    except ImportError:
        pytest.skip("PyTorch not installed, skipping torch dtype tests")


# Custom class for testing that properly triggers the first branch
class CustomObjWithQualName:
    pass


# Add __qualname__ directly to the class object
CustomObjWithQualName.__qualname__ = "CustomQualname"


# Test object with __qualname__ will be serialized with call=False
@patch("inspect.getmodule")
def test_safe_object_representer(mock_getmodule):
    """Test the safe object representer."""
    # Set up the mock to return a module with a name
    mock_module = SimpleNamespace(__name__="custom_module")
    mock_getmodule.return_value = mock_module

    # Create a dummy dumper
    dumper = yaml.SafeDumper(None)

    # Test case 1: We'll use our custom function object to ensure __qualname__ exists directly on the object
    # Create a callable function-like object that has __qualname__
    def test_func():
        pass

    obj = test_func

    # The first branch should be taken (_call_=False) since functions have __qualname__
    result = _safe_object_representer(dumper, obj)

    # Extract the values from the node
    mapping_data = {}
    for key_node, value_node in result.value:
        mapping_data[key_node.value] = value_node.value

    # Verify the result is using the first branch (call=False)
    assert "test_func" in mapping_data["_target_"]
    assert mapping_data["_call_"] == "false" or mapping_data["_call_"] is False

    # Test case 2: Regular class instance (falls back to __class__)
    class SimpleTestClass:
        pass

    obj = SimpleTestClass()

    # When serializing normal instances, _call_ should be True
    result = _safe_object_representer(dumper, obj)

    # Extract the values from the node
    mapping_data = {}
    for key_node, value_node in result.value:
        mapping_data[key_node.value] = value_node.value

    # Verify it uses the object's class with _call_=True
    assert "SimpleTestClass" in mapping_data["_target_"]
    assert mapping_data["_call_"] == "true" or mapping_data["_call_"] is True


def test_full_yaml_dump():
    """Test a complete YAML dump with various object types."""

    def local_function():
        pass

    # Create test data with various types
    test_data = {
        "function": dummy_function,
        "method": DummyClass.test_method,
        "static_method": DummyClass.static_method,
        "class_method": DummyClass.class_method,
        "lambda": lambda x: x * 2,
        "class_instance": DummyClass(),
        "nested": {
            "function": local_function,
        },
    }

    # Try to add torch dtype if available
    try:
        import torch

        test_data["dtype"] = torch.float32
    except ImportError:
        pass

    # Dump to YAML using our context manager
    with safe_yaml_representers():
        yaml_str = yaml.safe_dump(test_data)

    # Verify the result
    assert "_target_" in yaml_str
    assert "test_yaml_utils.DummyClass" in yaml_str
    assert "function" in yaml_str
    assert "method" in yaml_str
    assert "_call_: false" in yaml_str
    assert "_call_: true" in yaml_str  # For class instance


def test_serialize_methods():
    """Test serialization of different method types."""
    # Create test methods
    instance_method = DummyClass().test_method
    static_method = DummyClass.static_method
    class_method = DummyClass.class_method

    # Dump to YAML using our context manager
    with safe_yaml_representers():
        instance_yaml = yaml.safe_dump(instance_method)
        static_yaml = yaml.safe_dump(static_method)
        class_yaml = yaml.safe_dump(class_method)

    # Verify the results
    assert "test_method" in instance_yaml
    assert "_call_: false" in instance_yaml

    # These assertions will check for the method name in the target
    assert "static_method" in static_yaml
    assert "_call_: false" in static_yaml

    assert "class_method" in class_yaml
    assert "_call_: false" in class_yaml


def test_torch_module_not_found():
    """Test torch module not found branch."""
    # Save the original state of yaml.SafeDumper
    original_representers = yaml.SafeDumper.yaml_representers.copy()

    try:
        # Mock an import error for torch
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ModuleNotFoundError("Mocked torch not found")
            return original_import(name, *args, **kwargs)

        # Use the mock
        builtins.__import__ = mock_import

        # Call the context manager which should attempt to import torch
        with safe_yaml_representers():
            pass

        # If we get here, the ModuleNotFoundError was properly caught

    finally:
        # Restore the original import function
        builtins.__import__ = original_import
        # Restore the original representers
        yaml.SafeDumper.yaml_representers = original_representers


def test_torch_dtype_representer_direct():
    """Test torch dtype representer directly."""
    try:
        import torch

        # Create a mock dumper that will record the calls
        class MockDumper:
            def __init__(self):
                self.represented_data = None

            def represent_data(self, data):
                self.represented_data = data
                return data

        dumper = MockDumper()
        dtype = torch.float32

        # Call the representer directly
        _torch_dtype_representer(dumper, dtype)

        # Verify the call to represent_data
        assert dumper.represented_data is not None
        assert dumper.represented_data["_target_"] == str(dtype)
        assert dumper.represented_data["_call_"] is False

    except ImportError:
        pytest.skip("PyTorch not installed, skipping torch dtype tests")


@patch("nemo.tron.utils.yaml_utils._safe_object_representer")
def test_safe_object_representer_edge_cases(mock_representer):
    """Test edge cases in the safe_object_representer function."""
    # Create a dummy dumper
    dumper = yaml.SafeDumper(None)

    # Create a test object
    class CustomObj:
        pass

    obj = CustomObj()

    # Mock the implementation to return a valid result
    mock_representer.return_value = dumper.represent_data(
        {"_target_": "tests.tron.utils.test_yaml_utils_utils.CustomObj", "_call_": True}
    )

    # Run this inside the safe_yaml_representers context
    with safe_yaml_representers():
        # This should use our mocked function
        result = yaml.safe_dump(obj)

        # The call to our mocked function should happen
        mock_representer.assert_called_once()

        # Verify the output
        assert "CustomObj" in result
        assert "_call_: true" in result


def test_custom_safe_yaml_representers():
    """Test registering custom representers inside the context manager."""

    # Create a custom class
    class CustomClass:
        pass

    # Test with a custom representer
    def custom_representer(dumper, data):
        value = {"_special_": "custom"}
        return dumper.represent_data(value)

    with safe_yaml_representers():
        # Add our custom representer inside the context
        yaml.SafeDumper.add_representer(CustomClass, custom_representer)

        # Test the custom representer
        result = yaml.safe_dump(CustomClass())
        assert "_special_: custom" in result

    # Verify our custom representer was removed
    with pytest.raises(yaml.representer.RepresenterError):
        yaml.safe_dump(CustomClass())


def test_partial_representer():
    """Test the serialization of partial objects to YAML and back."""
    # Test with a simple partial function without args
    simple_partial = partial(dummy_function)

    # Test with a partial function with args and kwargs
    complex_partial = partial(dummy_function_with_args, 10, c=30)

    # Test with a method
    method_partial = partial(DummyClass.test_method)

    with safe_yaml_representers():
        # Test simple partial
        yaml_str = yaml.safe_dump(simple_partial)
        loaded_data = yaml.safe_load(yaml_str)

        # Verify the loaded data structure
        assert "_target_" in loaded_data
        assert loaded_data["_target_"] == "utils.test_yaml_utils.dummy_function"
        assert loaded_data["_partial_"] is True
        assert "_args_" in loaded_data
        assert loaded_data["_args_"] == []

        # Test complex partial with args and kwargs
        yaml_str = yaml.safe_dump(complex_partial)
        loaded_data = yaml.safe_load(yaml_str)

        # Verify the complex partial structure
        assert "_target_" in loaded_data
        assert loaded_data["_target_"] == "utils.test_yaml_utils.dummy_function_with_args"
        assert loaded_data["_partial_"] is True
        assert "_args_" in loaded_data
        assert loaded_data["_args_"] == [10]
        assert "c" in loaded_data
        assert loaded_data["c"] == 30

        # Test method partial
        yaml_str = yaml.safe_dump(method_partial)
        loaded_data = yaml.safe_load(yaml_str)

        # Verify the method partial structure
        assert "_target_" in loaded_data
        assert loaded_data["_target_"] == "utils.test_yaml_utils.DummyClass.test_method"
        assert loaded_data["_partial_"] is True
        assert "_args_" in loaded_data
        assert loaded_data["_args_"] == []


def test_full_yaml_dump_with_partial():
    """Test YAML dump with partial objects."""
    # Create test data with partial functions
    test_data = {
        "simple_partial": partial(dummy_function),
        "complex_partial": partial(dummy_function_with_args, 10, c=30),
        "method_partial": partial(DummyClass.test_method),
    }

    # Dump to YAML using our context manager
    with safe_yaml_representers():
        yaml_str = yaml.safe_dump(test_data)

    # Verify the result contains partial information
    assert "_target_" in yaml_str
    assert "_partial_: true" in yaml_str
    assert "test_yaml_utils.dummy_function" in yaml_str
    assert "test_yaml_utils.dummy_function_with_args" in yaml_str
    assert "c: 30" in yaml_str
    assert "test_yaml_utils.DummyClass.test_method" in yaml_str
