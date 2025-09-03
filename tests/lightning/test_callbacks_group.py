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

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from lightning.pytorch.callbacks import Callback as PTLCallback

from nemo.lightning.base_callback import BaseCallback
from nemo.lightning.callback_group import CallbackGroup, hook_class_init_with_callbacks


def _fresh_group_module():
    """Import the callback_group module from nemo.lightning."""
    if 'nemo.lightning.callback_group' in sys.modules:
        del sys.modules['nemo.lightning.callback_group']
    # Stub out OneLoggerNeMoCallback to avoid importing external deps during tests
    stub_one_logger_mod = types.ModuleType('nemo.lightning.one_logger_callback')

    class _StubOneLoggerCallback(BaseCallback):
        def __init__(self, *args, **kwargs):
            pass

        def update_config(self, *args, **kwargs):
            pass

    setattr(stub_one_logger_mod, 'OneLoggerNeMoCallback', _StubOneLoggerCallback)

    with patch.dict(sys.modules, {'nemo.lightning.one_logger_callback': stub_one_logger_mod}):
        return importlib.import_module('nemo.lightning.callback_group')


def test_base_callback_noops_do_not_raise():
    """Test BaseCallback hooks are no-ops and do not raise exceptions."""
    cb = BaseCallback()

    cb.on_app_start()
    cb.on_app_end()
    cb.on_model_init_start()
    cb.on_model_init_end()
    cb.on_dataloader_init_start()
    cb.on_dataloader_init_end()
    cb.on_optimizer_init_start()
    cb.on_optimizer_init_end()
    cb.on_load_checkpoint_start()
    cb.on_load_checkpoint_end()
    cb.on_save_checkpoint_start()
    cb.on_save_checkpoint_end()
    cb.on_save_checkpoint_success()
    cb.update_config()


def test_base_callback_is_ptl_callback():
    """Test BaseCallback derives from Lightning PTL Callback."""
    assert isinstance(BaseCallback(), PTLCallback)


def test_callback_group_singleton_identity():
    """Test CallbackGroup returns the same singleton instance."""
    mod = _fresh_group_module()
    a = mod.CallbackGroup.get_instance()
    b = mod.CallbackGroup.get_instance()
    assert a is b


def test_callback_group_update_config_fanout_and_attach(monkeypatch):
    """Test update_config fans out to callbacks and attaches them to trainer."""
    mod = _fresh_group_module()
    group = mod.CallbackGroup.get_instance()

    mock_cb = MagicMock()
    group._callbacks = [mock_cb]

    class Trainer:
        def __init__(self):
            self.callbacks = []

    trainer = Trainer()
    marker = object()
    group.update_config('v2', trainer, data=marker)

    assert mock_cb.update_config.called
    kwargs = mock_cb.update_config.call_args.kwargs
    assert kwargs['nemo_version'] == 'v2'
    assert kwargs['trainer'] is trainer
    assert kwargs['data'] is marker
    assert trainer.callbacks[0] is mock_cb


def test_callback_group_dynamic_dispatch_calls_when_present():
    """Test dynamic dispatch calls methods when present on callbacks."""
    mod = _fresh_group_module()
    group = mod.CallbackGroup.get_instance()

    mock_cb = MagicMock()
    group._callbacks = [mock_cb]

    group.on_app_start()
    assert mock_cb.on_app_start.called


def test_callback_group_dynamic_dispatch_ignores_missing_methods():
    """Test dynamic dispatch ignores missing methods without raising."""
    mod = _fresh_group_module()
    group = mod.CallbackGroup.get_instance()

    class Dummy:
        pass

    group._callbacks = [Dummy()]

    # Should not raise even if method not present
    group.on_nonexistent_method()


def test_hook_class_init_with_callbacks_wraps_and_emits(monkeypatch):
    """Test hook wraps __init__ and emits start/end callbacks once."""
    mod = _fresh_group_module()
    group = mod.CallbackGroup.get_instance()

    start = MagicMock()
    end = MagicMock()

    monkeypatch.setattr(group, 'on_model_init_start', start)
    monkeypatch.setattr(group, 'on_model_init_end', end)

    class Dummy:
        def __init__(self):
            self.x = 1

    mod.hook_class_init_with_callbacks(Dummy, 'on_model_init_start', 'on_model_init_end')

    d = Dummy()
    assert d.x == 1
    assert start.called
    assert end.called
    # Flag indicating wrapping applied
    assert getattr(Dummy.__init__, '_callback_group_wrapped', False) is True


def test_hook_class_init_with_callbacks_idempotent():
    """Test hook is idempotent and does not re-wrap on repeated calls."""
    mod = _fresh_group_module()

    class Dummy:
        def __init__(self):
            pass

    mod.hook_class_init_with_callbacks(Dummy, 'on_model_init_start', 'on_model_init_end')
    first = Dummy.__init__
    mod.hook_class_init_with_callbacks(Dummy, 'on_model_init_start', 'on_model_init_end')
    second = Dummy.__init__
    assert first is second
