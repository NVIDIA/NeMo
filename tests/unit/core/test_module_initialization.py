# ! /usr/bin/python
# -*- coding: utf-8 -*-

# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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

from unittest import TestCase

import pytest

from nemo.backends.pytorch.nm import TrainableNM
from nemo.backends.pytorch.tutorials import TaylorNet
from nemo.core.neural_modules import NmTensor
from nemo.core.neural_types import ChannelType, NeuralType


@pytest.mark.usefixtures("neural_factory")
class ModuleInitializationTestCase(TestCase):
    class TestNM1(TrainableNM):
        def __init__(self, var1=1, var2=2, var3=3):
            super().__init__()

    class TestNM2(TestNM1):
        def __init__(self, var2):
            super().__init__(var2=var2)

    def setUp(self) -> None:
        super().setUp()

        # Mockup abstract methods.
        ModuleInitializationTestCase.TestNM1.__abstractmethods__ = set()
        ModuleInitializationTestCase.TestNM2.__abstractmethods__ = set()

    @pytest.mark.unit
    def test_default_init_params(self):
        simple_nm = ModuleInitializationTestCase.TestNM1(var1=1)
        init_params = simple_nm.init_params
        self.assertEqual(init_params["var1"], 1)
        self.assertEqual(init_params["var2"], 2)
        self.assertEqual(init_params["var3"], 3)

    @pytest.mark.unit
    def test_simple_init_params(self):
        simple_nm = ModuleInitializationTestCase.TestNM1(var1=10, var3=30)
        init_params = simple_nm.init_params
        self.assertEqual(init_params["var1"], 10)
        self.assertEqual(init_params["var2"], 2)
        self.assertEqual(init_params["var3"], 30)

    @pytest.mark.unit
    def test_nested_init_params(self):
        simple_nm = ModuleInitializationTestCase.TestNM2(var2="hello")
        init_params = simple_nm.init_params
        self.assertEqual(init_params["var2"], "hello")

    @pytest.mark.unit
    def test_constructor_TaylorNet(self):
        tn = TaylorNet(dim=4)
        self.assertEqual(tn.init_params["dim"], 4)

    @pytest.mark.unit
    def test_call_TaylorNet(self):
        x_tg = NmTensor(
            producer=None,
            producer_args=None,
            name=None,
            ntype=NeuralType(elements_type=ChannelType(), axes=('B', 'D')),
        )

        tn = TaylorNet(dim=4)
        # note that real port's name: x was used
        y_pred = tn(x=x_tg)
        self.assertEqual(y_pred.producer, tn)
        self.assertEqual(y_pred.producer_args.get("x"), x_tg)
