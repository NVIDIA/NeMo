# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from nemo.utils.import_utils import UnavailableError, UnavailableMeta, is_unavailable, safe_import, safe_import_from


class TestSafeImport:
    def test_successful_import(self):
        """Test safe_import with a module that exists."""
        module, success = safe_import("os")
        assert success is True
        assert isinstance(module, types.ModuleType)
        assert module.__name__ == "os"

    def test_failed_import(self):
        """Test safe_import with a module that doesn't exist."""
        module, success = safe_import("nonexistent_module")
        assert success is False
        assert is_unavailable(module)
        assert type(module) is UnavailableMeta

    def test_import_with_custom_message(self):
        """Test safe_import with a custom error message."""
        custom_msg = "Custom error message"
        module, success = safe_import("nonexistent_module", msg=custom_msg)

        assert success is False
        assert is_unavailable(module)

        # Verify the custom message is used when trying to use the module
        with pytest.raises(UnavailableError, match=custom_msg):
            module()

    def test_import_with_alternative(self):
        """Test safe_import with an alternative module."""
        alt_module = object()
        module, success = safe_import("nonexistent_module", alt=alt_module)

        assert success is False
        assert module is alt_module

    def test_unavailable_module_raises_error_when_used(self):
        """Test that using a UnavailableMeta placeholder raises UnavailableError."""
        module, success = safe_import("nonexistent_module")

        assert success is False

        # Test various operations that should raise UnavailableError
        with pytest.raises(UnavailableError):
            module()

        with pytest.raises(UnavailableError):
            module.attribute

        with pytest.raises(UnavailableError):
            module + 1

        with pytest.raises(UnavailableError):
            module == 1


class TestSafeImportFrom:
    def test_successful_import_from(self):
        """Test safe_import_from with a symbol that exists."""
        symbol, success = safe_import_from("os", "path")
        assert success is True
        assert symbol is os.path

    def test_failed_import_from_nonexistent_module(self):
        """Test safe_import_from with a module that doesn't exist."""
        symbol, success = safe_import_from("nonexistent_module", "nonexistent_symbol")
        assert success is False
        assert is_unavailable(symbol)

    def test_failed_import_from_nonexistent_symbol(self):
        """Test safe_import_from with a symbol that doesn't exist in an existing module."""
        symbol, success = safe_import_from("os", "nonexistent_symbol")
        assert success is False
        assert is_unavailable(symbol)

    def test_import_from_with_custom_message(self):
        """Test safe_import_from with a custom error message."""
        custom_msg = "Custom error message for symbol"
        symbol, success = safe_import_from("os", "nonexistent_symbol", msg=custom_msg)

        assert success is False

        # Verify the custom message is used when trying to use the symbol
        with pytest.raises(UnavailableError, match=custom_msg):
            symbol()

    def test_import_from_with_alternative(self):
        """Test safe_import_from with an alternative symbol."""
        alt_symbol = object()
        symbol, success = safe_import_from("os", "nonexistent_symbol", alt=alt_symbol)

        assert success is False
        assert symbol is alt_symbol

    def test_fallback_module(self):
        """Test safe_import_from with a fallback module."""
        # First import fails, but fallback succeeds
        with patch('importlib.import_module') as mock_import:
            # Mock the first import to fail
            def side_effect(name):
                if name == "primary_module":
                    raise ImportError("Module not found")
                elif name == "fallback_module":
                    mock_module = MagicMock()
                    mock_module.symbol = "fallback_symbol"
                    return mock_module
                else:
                    raise ImportError(f"Unexpected module: {name}")

            mock_import.side_effect = side_effect

            symbol, success = safe_import_from("primary_module", "symbol", fallback_module="fallback_module")

            assert success is True
            assert symbol == "fallback_symbol"

    def test_fallback_module_both_fail(self):
        """Test safe_import_from when both primary and fallback modules fail."""
        symbol, success = safe_import_from("nonexistent_primary", "symbol", fallback_module="nonexistent_fallback")

        assert success is False
        assert is_unavailable(symbol)
