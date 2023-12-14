# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from inspect import isclass
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

import nemo.core.neural_types.elements as nelements
from nemo.core.classes import Exportable, NeuralModule, typecheck
from nemo.core.neural_types import FloatType, NeuralType, VoidType


class SimpleLinear(NeuralModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20, bias=False)

    def forward(self, x):
        return self.linear(x)


class SimpleLinearExportable(NeuralModule, Exportable):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20, bias=False)

    def forward(self, x):
        return self.linear(x)


class SimpleLinearWithTypes(NeuralModule):
    @property
    def input_types(self) -> Dict[str, Any]:
        return {
            "x": NeuralType(None, FloatType()),
        }

    @property
    def output_types(self) -> Dict[str, Any]:
        return {
            "output": NeuralType(None, VoidType()),
        }

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20, bias=False)

    @typecheck()
    def forward(self, x):
        return self.linear(x)


class DummyModuleWithIOTypes(NeuralModule):
    @property
    def input_types(self) -> Dict[str, NeuralType]:
        return {
            "x": NeuralType(None, VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        return {
            "output": NeuralType(None, FloatType()),
        }

    def forward(self, x):
        return x


class TestTorchJitCompatibility:
    @pytest.mark.unit
    def test_simple_linear(self):
        module = torch.jit.script(SimpleLinear())
        x = torch.zeros(2, 10)
        result = module(x)
        assert result.shape == (2, 20)
        assert torch.allclose(result, torch.zeros_like(result))

    @pytest.mark.unit
    def test_simple_linear_exportable(self):
        module = torch.jit.script(SimpleLinearExportable())
        x = torch.zeros(2, 10)
        result = module(x)
        assert result.shape == (2, 20)
        assert torch.allclose(result, torch.zeros_like(result))

    @pytest.mark.unit
    def test_simple_linear_with_types(self):
        module = torch.jit.script(SimpleLinearWithTypes())
        x = torch.zeros(2, 10)
        result = module(x)
        assert result.shape == (2, 20)
        assert torch.allclose(result, torch.zeros_like(result))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "neural_type",
        [
            getattr(nelements, neural_type_str)
            for neural_type_str in dir(nelements)
            if isclass(getattr(nelements, neural_type_str))
            and issubclass(getattr(nelements, neural_type_str), nelements.ElementType)
            and not (getattr(nelements, neural_type_str) is nelements.ElementType)
        ],
    )
    def test_element_compilable(self, neural_type):
        """
        Tests that NeuralType elements are compilable by TorchScript.
        """

        @torch.jit.script
        def identity(x: torch.Tensor):
            if isinstance(neural_type(), nelements.VoidType):
                return x
            else:
                return x + 1

        _ = identity(torch.tensor(1.0))

    @pytest.mark.unit
    def test_dummy_module_with_io_types(self):
        module = torch.jit.script(DummyModuleWithIOTypes())
        x = torch.rand(2, 10)
        result = module(x)
        assert result.shape == x.shape
        assert torch.allclose(result, x)

    @pytest.mark.unit
    def test_chain_with_types(self):
        dummy_module = torch.jit.script(DummyModuleWithIOTypes())
        module = torch.jit.script(SimpleLinearWithTypes())
        x = torch.zeros(2, 10)
        result = module(dummy_module(x))
        assert result.shape == (2, 20)
        assert torch.allclose(result, torch.zeros_like(result))
