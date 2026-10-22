# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import builtins

import pytest

from nemo.collections.asr.parts.utils.wfst_utils import _is_notebook


def _fake_shell(class_name: str):
    shell_cls = type(class_name, (), {})
    return lambda: shell_cls()


@pytest.mark.unit
@pytest.mark.parametrize(
    "shell_name, expected",
    [
        ("ZMQInteractiveShell", True),  # Jupyter notebook / qtconsole
        ("Shell", True),  # Google Colab
        ("TerminalInteractiveShell", False),  # IPython in a terminal
        ("SomeOtherShell", False),
    ],
)
def test_is_notebook(monkeypatch, shell_name, expected):
    monkeypatch.setattr(builtins, "get_ipython", _fake_shell(shell_name), raising=False)
    assert _is_notebook() is expected


@pytest.mark.unit
def test_is_notebook_plain_python(monkeypatch):
    monkeypatch.delattr(builtins, "get_ipython", raising=False)
    assert _is_notebook() is False
