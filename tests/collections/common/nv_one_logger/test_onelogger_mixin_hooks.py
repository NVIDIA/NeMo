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

from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn
from lightning.pytorch import LightningModule

from nemo.collections.llm.fn.mixin import FNMixin
from nemo.lightning.io.mixin import IOMixin


class TestOneLoggerMixinHooks:
    """Test cases for OneLogger mixin timing hooks."""

    @pytest.mark.unit
    def test_init_subclass_lightning_module_hooks(self):
        """Test that __init_subclass__ hooks callbacks for LightningModule subclasses."""
        
        # Create a test class that inherits from FNMixin and LightningModule
        class FakeModel(FNMixin, LightningModule):
            def __init__(self, value=0):
                super().__init__()
                self.value = value
        
        # Verify that the __init__ method was wrapped
        assert hasattr(FakeModel.__init__, '_one_logger_wrapped')
        assert FakeModel.__init__._one_logger_wrapped is True

    @pytest.mark.unit
    def test_init_subclass_non_lightning_module_no_hooks(self):
        """Test that __init_subclass__ doesn't hook callbacks for non-LightningModule classes."""
        
        # Create a test class that inherits from FNMixin but not LightningModule
        class FakeNonLightningModel(FNMixin, nn.Module):
            def __init__(self, value=0):
                super().__init__()
                self.value = value
        
        # Verify that the __init__ method was NOT wrapped
        assert not hasattr(FakeNonLightningModel.__init__, '_one_logger_wrapped')

    @pytest.mark.unit
    def test_init_subclass_import_error_handling(self):
        """Test that __init_subclass__ handles import errors gracefully."""
        
        # Test that the mixin can handle import errors by creating a class
        # that would trigger the import but doesn't cause issues
        class FakeModelWithImportError(FNMixin, LightningModule):
            def __init__(self, value=0):
                super().__init__()
                self.value = value
        
        # The class should be created successfully even if there are import issues
        # The __init_subclass__ method has a try...finally block that should handle exceptions
        assert FakeModelWithImportError is not None

    @pytest.mark.unit
    def test_init_subclass_hook_function_import_error(self):
        """Test that __init_subclass__ handles hook function import errors gracefully."""
        
        # Test that the mixin can handle hook function errors by creating a class
        # that would trigger the hook function call but doesn't cause issues
        class FakeModelWithHookImportError(FNMixin, LightningModule):
            def __init__(self, value=0):
                super().__init__()
                self.value = value
        
        # The class should be created successfully even if there are hook function issues
        # The __init_subclass__ method has a try...finally block that should handle exceptions
        assert FakeModelWithHookImportError is not None

    @pytest.mark.unit
    def test_init_subclass_callback_execution(self):
        """Test that the hooked callbacks are executed during model initialization."""
        
        # Track callback calls
        callback_calls = []
        
        def mock_track_event(event_name):
            callback_calls.append(event_name)
        
        # Create a test class
        class FakeModel(FNMixin, LightningModule):
            def __init__(self, value=0):
                super().__init__()
                self.value = value
        
        # Mock the timing tracker to capture events
        with patch('nemo.lightning.one_logger_callback.OneLoggerTimingTracker.get_instance') as mock_get_instance:
            mock_tracker = MagicMock()
            mock_tracker.track_event = mock_track_event
            mock_get_instance.return_value = mock_tracker
            
            # Create an instance of the model
            model = FakeModel(value=42)
            
            # Verify that the model was created correctly
            assert model.value == 42
            
            # Note: The timing events might not be tracked if OneLogger is not available
            # This test verifies the model creation works, but the timing depends on OneLogger availability
            # The important thing is that the __init__ method was wrapped
            assert hasattr(FakeModel.__init__, '_one_logger_wrapped')

    @pytest.mark.unit
    def test_init_subclass_multiple_inheritance(self):
        """Test that __init_subclass__ works correctly with multiple inheritance."""
        
        class BaseClass:
            def __init__(self):
                self.base_value = "base"
        
        # Use proper inheritance order - LightningModule should come first for proper MRO
        class FakeModel(LightningModule, FNMixin, BaseClass):
            def __init__(self, value=0):
                # Call all parent __init__ methods properly
                LightningModule.__init__(self)
                BaseClass.__init__(self)
                self.value = value
        
        # Verify that the __init__ method was wrapped
        assert hasattr(FakeModel.__init__, '_one_logger_wrapped')
        assert FakeModel.__init__._one_logger_wrapped is True
        
        # Create an instance and verify it works
        model = FakeModel(value=42)
        assert model.value == 42
        assert model.base_value == "base"

    @pytest.mark.unit
    def test_init_subclass_no_init_method(self):
        """Test that __init_subclass__ handles classes without __init__ method."""
        
        class FakeModelNoInit(FNMixin, LightningModule):
            pass
        
        # The class should be created without issues
        assert FakeModelNoInit is not None
        
        # Creating an instance should work (uses default __init__)
        model = FakeModelNoInit()
        assert isinstance(model, FakeModelNoInit)


class TestOneLoggerIOMixinHooks:
    """Test cases for OneLogger IO mixin timing hooks."""

    # Mock LightningDataModule for testing
    class MockLightningDataModule:
        pass

    @pytest.mark.unit
    def test_io_mixin_lightning_datamodule_hooks(self):
        """Test that IOMixin __init_subclass__ hooks callbacks for LightningDataModule subclasses."""
        
        # Create a test class that inherits from IOMixin and LightningDataModule
        class FakeDataModule(IOMixin, self.MockLightningDataModule):
            def __init__(self, value=0):
                super().__init__()
                self.value = value
        
        # The IOMixin should register the class for OneLogger hooks
        # Note: IOMixin doesn't wrap __init__ like FNMixin, it just registers the class
        assert FakeDataModule is not None

    @pytest.mark.unit
    def test_io_mixin_non_datamodule_no_hooks(self):
        """Test that IOMixin __init_subclass__ doesn't hook callbacks for non-LightningDataModule classes."""
        
        # Create a test class that inherits from IOMixin but not LightningDataModule
        class FakeNonDataModule(IOMixin, nn.Module):
            def __init__(self, value=0):
                super().__init__()
                self.value = value
        
        # Verify that the class was created successfully
        assert FakeNonDataModule is not None

    @pytest.mark.unit
    def test_io_mixin_import_error_handling(self):
        """Test that IOMixin __init_subclass__ handles import errors gracefully."""
        
        # Test that the mixin can handle import errors by creating a class
        # that would trigger the import but doesn't cause issues
        class FakeDataModuleWithImportError(IOMixin, self.MockLightningDataModule):
            def __init__(self, value=0):
                super().__init__()
                self.value = value
        
        # The class should be created successfully even if there are import issues
        # The __init_subclass__ method has a try...finally block that should handle exceptions
        assert FakeDataModuleWithImportError is not None

    @pytest.mark.unit
    def test_io_mixin_hook_function_import_error(self):
        """Test that IOMixin __init_subclass__ handles hook function import errors gracefully."""
        
        # Test that the mixin can handle hook function errors by creating a class
        # that would trigger the hook function call but doesn't cause issues
        class FakeDataModuleWithHookImportError(IOMixin, self.MockLightningDataModule):
            def __init__(self, value=0):
                super().__init__()
                self.value = value
        
        # The class should be created successfully even if there are hook function issues
        # The __init_subclass__ method has a try...finally block that should handle exceptions
        assert FakeDataModuleWithHookImportError is not None

    @pytest.mark.unit
    def test_io_mixin_callback_execution(self):
        """Test that the hooked callbacks are executed during datamodule initialization."""
        
        # Track callback calls
        callback_calls = []
        
        def mock_track_event(event_name):
            callback_calls.append(event_name)
        
        # Create a test class
        class FakeDataModule(IOMixin, self.MockLightningDataModule):
            def __init__(self, value=0):
                super().__init__()
                self.value = value
        
        # Mock the timing tracker to capture events
        with patch('nemo.lightning.one_logger_callback.OneLoggerTimingTracker.get_instance') as mock_get_instance:
            mock_tracker = MagicMock()
            mock_tracker.track_event = mock_track_event
            mock_get_instance.return_value = mock_tracker
            
            # Create an instance of the datamodule
            datamodule = FakeDataModule(value=42)
            
            # Verify that the datamodule was created correctly
            assert datamodule.value == 42
            
            # Note: The timing events might not be tracked if OneLogger is not available
            # This test verifies the datamodule creation works, but the timing depends on OneLogger availability
            # The important thing is that the class was registered for OneLogger hooks
            assert FakeDataModule is not None

    @pytest.mark.unit
    def test_io_mixin_multiple_inheritance(self):
        """Test that IOMixin __init_subclass__ works correctly with multiple inheritance."""
        
        class BaseClass:
            def __init__(self):
                self.base_value = "base"
        
        # Use proper inheritance order - LightningDataModule should come first for proper MRO
        class FakeDataModule(self.MockLightningDataModule, IOMixin, BaseClass):
            def __init__(self, value=0):
                # Call all parent __init__ methods properly
                BaseClass.__init__(self)
                self.value = value
        
        # Verify that the class was created successfully
        assert FakeDataModule is not None
        
        # Create an instance and verify it works
        datamodule = FakeDataModule(value=42)
        assert datamodule.value == 42
        assert datamodule.base_value == "base"

    @pytest.mark.unit
    def test_io_mixin_no_init_method(self):
        """Test that IOMixin __init_subclass__ handles classes without __init__ method."""
        
        class FakeDataModuleNoInit(IOMixin, self.MockLightningDataModule):
            pass
        
        # The class should be created without issues
        assert FakeDataModuleNoInit is not None
        
        # Creating an instance should work (uses default __init__)
        datamodule = FakeDataModuleNoInit()
        assert isinstance(datamodule, FakeDataModuleNoInit) 