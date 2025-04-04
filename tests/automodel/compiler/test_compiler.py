# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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


import re
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo.automodel.compiler import (
    TorchCompileConfig,
    compile_module,
    compile_module_from_config,
    extract_module_attr_name,
    get_modules_from_selector,
    listify,
)


def test_extract_module_attr_name_with_module():
    mock_pl_module = MagicMock(spec=[])
    mock_pl_module.module = MagicMock()
    assert extract_module_attr_name(mock_pl_module) == 'module', mock_pl_module


def test_extract_module_attr_name_with_model():
    mock_pl_module = MagicMock(spec=[])
    mock_pl_module.model = MagicMock()
    assert extract_module_attr_name(mock_pl_module) == 'model', mock_pl_module


def test_extract_module_attr_name_raises():
    mock_pl_module = MagicMock(spec=[])
    # no 'module' or 'model'
    with pytest.raises(ValueError, match="Expected lightning_module to have a .model or .module"):
        extract_module_attr_name(mock_pl_module)


def test_listify_non_list():
    assert listify("test") == ["test"]


def test_listify_list():
    assert listify(["test"]) == ["test"]


def test_get_modules_from_selector_none_selector():
    model = MagicMock()
    collected = list(get_modules_from_selector(model, None))
    assert collected == [model]


def test_get_modules_from_selector_empty_string():
    model = MagicMock()
    collected = list(get_modules_from_selector(model, ""))
    assert collected == [model]


def test_get_modules_from_selector_star():
    model = MagicMock()
    collected = list(get_modules_from_selector(model, "*"))
    assert collected == [model]


def test_get_modules_from_selector_exact_path():
    # Example: model.encoder.layer
    child_module = torch.nn.Linear(3, 3)
    parent_module = torch.nn.Module()
    parent_module.encoder = torch.nn.Module()
    parent_module.encoder.layer = child_module

    collected = list(get_modules_from_selector(parent_module, "encoder.layer"))
    assert collected == [child_module]


def test_get_modules_from_selector_non_existent_attr():
    parent_module = torch.nn.Module()
    parent_module.encoder = torch.nn.Module()
    with pytest.raises(AttributeError, match="has no attribute"):
        list(get_modules_from_selector(parent_module, "decoder"))


def test_get_modules_from_selector_attr_is_not_module():
    parent_module = torch.nn.Module()
    parent_module.something = "I am not a module"
    with pytest.raises(AttributeError, match="is not an nn.Module"):
        list(get_modules_from_selector(parent_module, "something"))


def test_get_modules_from_selector_wildcard_children():
    parent_module = torch.nn.Module()
    parent_module.block1 = torch.nn.Linear(3, 3)
    parent_module.block2 = torch.nn.Linear(3, 3)
    collected = list(get_modules_from_selector(parent_module, "block*"))
    assert len(collected) == 2


def test_jit_config_assertion():
    # Should raise if not TorchCompileConfig / ThunderConfig
    mock_module = MagicMock()
    with pytest.raises(ValueError):
        compile_module({}, mock_module)


@pytest.fixture
def mock_module_torch():
    # Create a mock that pretends to be an nn.Module
    pl_module = MagicMock(spec=torch.nn.Module)
    # Make sure there's a .model attribute that we can access
    pl_module.model = MagicMock(spec=torch.nn.Module)
    # Initially, pretend it is not yet compiled
    pl_module._compiled = False
    return pl_module


def test_compile_module_torch(mock_module_torch):
    config = TorchCompileConfig(kwargs={"some_arg": 123})
    compile_module_from_config(config, mock_module_torch)
    mock_module_torch.compile.assert_called_once_with(some_arg=123)
    assert mock_module_torch._compiled == True


def test_compile_module_torch_with_path(mock_module_torch):
    config = TorchCompileConfig(module_selector="block1", kwargs={"some_arg": 123})

    # Ensure there's a `block1` attribute on the module so getattr(module, "block1") works
    mock_module_torch.block1 = MagicMock(spec=torch.nn.Module)

    with (
        patch("nemo.automodel.compiler.module_compiler.extract_module_attr_name", return_value="model"),
        patch(
            "nemo.automodel.compiler.module_compiler.get_modules_from_selector",
            return_value=[mock_module_torch.block1],
        ),
        patch("nemo.automodel.compiler.module_compiler.compile_module") as mock_compile,
    ):

        compile_module_from_config(config, mock_module_torch)
        mock_compile.assert_called_once_with(config, mock_module_torch.block1)

    # Now ensure _compiled was set to True
    assert mock_module_torch._compiled is True


def test_compile_module_none():
    mock_module = MagicMock()
    config = None
    with pytest.raises(ValueError):
        compile_module(config, mock_module)
    mock_module.compile.assert_not_called()


@pytest.mark.parametrize("config_class", [TorchCompileConfig])
def test_compile_sets_compiled_flag(config_class):
    # Arrange
    pl_module = MagicMock()
    # By default, pretend it's not compiled yet:
    setattr(pl_module, "_compiled", False)
    config = config_class()

    # Mock out dependencies
    with (
        patch("nemo.automodel.compiler.extract_module_attr_name", return_value="model"),
        patch("nemo.automodel.compiler.get_modules_from_selector", return_value=[MagicMock()]),
        patch("nemo.automodel.compiler.compile_module"),
    ):
        # Act
        compile_module_from_config(config, pl_module)

    # Assert
    assert getattr(pl_module, "_compiled") is True


def test_compile_does_not_set_compiled_when_config_is_none():
    # Arrange
    pl_module = MagicMock()
    setattr(pl_module, "_compiled", False)

    # Act
    compile_module_from_config(None, pl_module)

    # Assert
    assert getattr(pl_module, "_compiled") is False


def test_compile_skips_if_already_compiled():
    # Arrange
    pl_module = MagicMock()
    setattr(pl_module, "_compiled", True)
    config = TorchCompileConfig()

    with (
        patch("nemo.automodel.compiler.utils.extract_module_attr_name", return_value="model"),
        patch("nemo.automodel.compiler.get_modules_from_selector") as mock_selector,
    ):
        # Act
        compile_module_from_config(config, pl_module)

    # Assert: no further calls should happen if _compiled was True
    mock_selector.assert_not_called()
    assert getattr(pl_module, "_compiled") is True  # remains True, unchanged
