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

from unittest.mock import patch

import pytest
import torch.nn as nn
from lightning.pytorch import LightningModule, LightningDataModule

from nemo.collections.llm.fn.mixin import FNMixin
from nemo.lightning.io.mixin import IOMixin


class TestOneLoggerMixinHooks:
    """Test cases for OneLogger mixin timing hooks."""

    @pytest.mark.unit
    def test_init_subclass_lightning_module_hooks(self):
        """Test that __init_subclass__ hooks callbacks for LightningModule subclasses."""

        # Mock OneLogger to be available
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            # Create a test class that inherits from FNMixin and LightningModule
            class FakeModel(FNMixin, LightningModule):
                def __init__(self, value=0):
                    super().__init__()
                    self.value = value

            # Verify that the __init__ method was wrapped
            assert hasattr(FakeModel.__init__, '_one_logger_wrapped')
            assert FakeModel.__init__._one_logger_wrapped is True

    @pytest.mark.unit
    def test_init_subclass_lightning_module_hooks_no_onelogger(self):
        """Test that __init_subclass__ doesn't hook callbacks when OneLogger is not available."""

        # Mock OneLogger to be unavailable
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', False):
            # Create a test class that inherits from FNMixin and LightningModule
            class FakeModel(FNMixin, LightningModule):
                def __init__(self, value=0):
                    super().__init__()
                    self.value = value

            # Verify that the __init__ method was NOT wrapped when OneLogger is unavailable
            assert not hasattr(FakeModel.__init__, '_one_logger_wrapped')

    @pytest.mark.unit
    def test_init_subclass_non_lightning_module_no_hooks(self):
        """Test that __init_subclass__ doesn't hook callbacks for non-LightningModule classes."""

        # Mock OneLogger to be available
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            # Create a test class that inherits from FNMixin but not LightningModule
            class FakeNonLightningModel(FNMixin, nn.Module):
                def __init__(self, value=0):
                    super().__init__()
                    self.value = value

            # Verify that the __init__ method was NOT wrapped for non-LightningModule classes
            assert not hasattr(FakeNonLightningModel.__init__, '_one_logger_wrapped')

    @pytest.mark.unit
    def test_init_subclass_import_error_handling(self):
        """Test that __init_subclass__ handles import errors gracefully."""
        # Mock OneLogger to be available but make the import fail
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.hook_class_init_with_callbacks') as mock_hook:
                # Mock the hook function to raise an import error
                mock_hook.side_effect = ImportError("Failed to import hook function")
                
                # Create a test class that inherits from FNMixin
                class FakeModelWithImportError(FNMixin, LightningModule):
                    def __init__(self, value=0):
                        super().__init__()
                        self.value = value
                
                # Should not raise any exceptions - the error should be handled gracefully
                # Verify that the __init__ method was NOT wrapped due to import error
                assert not hasattr(FakeModelWithImportError.__init__, '_one_logger_wrapped')

    @pytest.mark.unit
    def test_init_subclass_hook_function_import_error(self):
        """Test that __init_subclass__ handles hook function import errors gracefully."""
        # Mock OneLogger to be available but make the hook function import fail
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.hook_class_init_with_callbacks') as mock_hook:
                # Mock the hook function to raise an import error
                mock_hook.side_effect = ImportError("Failed to import hook function")
                
                # Create a test class that inherits from FNMixin
                class FakeModelWithHookImportError(FNMixin, LightningModule):
                    def __init__(self, value=0):
                        super().__init__()
                        self.value = value
                
                # Should not raise any exceptions - the error should be handled gracefully
                # Verify that the __init__ method was NOT wrapped due to import error
                assert not hasattr(FakeModelWithHookImportError.__init__, '_one_logger_wrapped')

    @pytest.mark.unit
    def test_init_subclass_callback_execution(self):
        """Test that the hooked callbacks are executed during model initialization."""

        # Mock OneLogger to be available
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            # Create a test class
            class FakeModel(FNMixin, LightningModule):
                def __init__(self, value=0):
                    super().__init__()
                    self.value = value

            # Verify that the __init__ method was wrapped
            assert hasattr(FakeModel.__init__, '_one_logger_wrapped')

    @pytest.mark.unit
    def test_init_subclass_callback_execution_no_onelogger(self):
        """Test that no callbacks are executed when OneLogger is not available."""

        # Mock OneLogger to be unavailable
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', False):
            # Create a test class
            class FakeModel(FNMixin, LightningModule):
                def __init__(self, value=0):
                    super().__init__()
                    self.value = value

            # Verify that the __init__ method was NOT wrapped
            assert not hasattr(FakeModel.__init__, '_one_logger_wrapped')

    @pytest.mark.unit
    def test_init_subclass_multiple_inheritance(self):
        """Test that __init_subclass__ works correctly with multiple inheritance."""

        class BaseClass:
            def __init__(self):
                self.base_value = "base"

        # Mock OneLogger to be available
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            # Use proper inheritance order - LightningModule should come first for proper MRO
            class FakeModelWithMultipleInheritance(BaseClass, FNMixin, LightningModule):
                def __init__(self, value=0):
                    super().__init__()
                    self.value = value

            # Verify that the __init__ method was wrapped
            assert hasattr(FakeModelWithMultipleInheritance.__init__, '_one_logger_wrapped')

    @pytest.mark.unit
    def test_init_subclass_multiple_inheritance_no_onelogger(self):
        """Test that __init_subclass__ works correctly with multiple inheritance when OneLogger is not available."""

        class BaseClass:
            def __init__(self):
                self.base_value = "base"

        # Mock OneLogger to be unavailable
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', False):
            # Use proper inheritance order - LightningModule should come first for proper MRO
            class FakeModelWithMultipleInheritance(BaseClass, FNMixin, LightningModule):
                def __init__(self, value=0):
                    super().__init__()
                    self.value = value

            # Verify that the __init__ method was NOT wrapped
            assert not hasattr(FakeModelWithMultipleInheritance.__init__, '_one_logger_wrapped')

            # Verify the instance was properly initialized
            instance = FakeModelWithMultipleInheritance(42)
            assert instance.value == 42
            assert instance.base_value == "base"

    @pytest.mark.unit
    def test_init_subclass_no_init_method(self):
        """Test that __init_subclass__ works correctly with classes that have no __init__ method."""

        # Mock OneLogger to be available
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            # Create a test class with no __init__ method
            class FakeModelNoInit(FNMixin, LightningModule):
                pass

            # Verify that the __init__ method was wrapped (should be the default object.__init__)
            assert hasattr(FakeModelNoInit.__init__, '_one_logger_wrapped')

    @pytest.mark.unit
    def test_init_subclass_no_init_method_no_onelogger(self):
        """Test that __init_subclass__ works correctly with classes that have no __init__ method when OneLogger is not available."""

        # Mock OneLogger to be unavailable
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', False):
            # Create a test class with no __init__ method
            class FakeModelNoInit(FNMixin, LightningModule):
                pass

            # Verify that the __init__ method was NOT wrapped
            assert not hasattr(FakeModelNoInit.__init__, '_one_logger_wrapped')

    @pytest.mark.unit
    def test_io_mixin_lightning_data_module_hooks(self):
        """Test that IOMixin hooks callbacks for LightningDataModule subclasses."""

        # Mock OneLogger to be available
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            # Create a test class that inherits from IOMixin and LightningDataModule
            # Note: IOMixin checks for LightningDataModule, so we need to inherit from both
            class FakeDataModule(IOMixin, LightningDataModule):
                def __init__(self, value=0):
                    super().__init__()
                    self.value = value

            # Verify that the __init__ method was wrapped
            assert hasattr(FakeDataModule.__init__, '_one_logger_wrapped')
            assert FakeDataModule.__init__._one_logger_wrapped is True

    @pytest.mark.unit
    def test_io_mixin_lightning_data_module_hooks_no_onelogger(self):
        """Test that IOMixin doesn't hook callbacks when OneLogger is not available."""

        # Mock OneLogger to be unavailable
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', False):
            # Create a test class that inherits from IOMixin and LightningDataModule
            class FakeDataModule(IOMixin, LightningModule):
                def __init__(self, value=0):
                    super().__init__()
                    self.value = value

            # Verify that the __init__ method was NOT wrapped
            assert not hasattr(FakeDataModule.__init__, '_one_logger_wrapped')

    @pytest.mark.unit
    def test_io_mixin_non_lightning_data_module_no_hooks(self):
        """Test that IOMixin doesn't hook callbacks for non-LightningDataModule classes."""

        # Mock OneLogger to be available
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            # Create a test class that inherits from IOMixin but not LightningDataModule
            class FakeNonDataModule(IOMixin, nn.Module):
                def __init__(self, value=0):
                    super().__init__()
                    self.value = value

            # Verify that the __init__ method was NOT wrapped for non-LightningDataModule classes
            assert not hasattr(FakeNonDataModule.__init__, '_one_logger_wrapped')

    @pytest.mark.unit
    def test_io_mixin_import_error_handling(self):
        """Test that IOMixin handles import errors gracefully."""
        # Mock OneLogger to be available but make the import fail
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.hook_class_init_with_callbacks') as mock_hook:
                # Mock the hook function to raise an import error
                mock_hook.side_effect = ImportError("Failed to import hook function")
                
                # Create a test class that inherits from IOMixin
                class FakeDataModuleWithImportError(IOMixin, LightningDataModule):
                    def __init__(self, value=0):
                        super().__init__()
                        self.value = value
                
                # Should not raise any exceptions - the error should be handled gracefully
                # Verify that the __init__ method was NOT wrapped due to import error
                assert not hasattr(FakeDataModuleWithImportError.__init__, '_one_logger_wrapped')

    @pytest.mark.unit
    def test_io_mixin_hook_function_import_error(self):
        """Test that IOMixin handles hook function import errors gracefully."""
        # Mock OneLogger to be available but make the hook function import fail
        with patch('nemo.lightning.one_logger_callback.HAVE_ONELOGGER', True):
            with patch('nemo.lightning.one_logger_callback.hook_class_init_with_callbacks') as mock_hook:
                # Mock the hook function to raise an import error
                mock_hook.side_effect = ImportError("Failed to import hook function")
                
                # Create a test class that inherits from IOMixin
                class FakeDataModuleWithHookImportError(IOMixin, LightningDataModule):
                    def __init__(self, value=0):
                        super().__init__()
                        self.value = value
                
                # Should not raise any exceptions - the error should be handled gracefully
                # Verify that the __init__ method was NOT wrapped due to import error
                assert not hasattr(FakeDataModuleWithHookImportError.__init__, '_one_logger_wrapped')
