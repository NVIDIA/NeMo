# ! /usr/bin/python
# -*- coding: utf-8 -*-

# =============================================================================
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
# =============================================================================

from unittest import TestCase

import pytest

import nemo


@pytest.mark.usefixtures("neural_factory")
class TestNeuralGraph(TestCase):
    @pytest.mark.integration
    def test_create_simple_graph(self):
        # Create modules.
        dl = nemo.tutorials.RealFunctionDataLayer(n=100, batch_size=16)
        fx = nemo.tutorials.TaylorNet(dim=4)
        loss = nemo.tutorials.MSELoss()

        # Create the graph by connnecting the modules.
        x, y = dl()
        y_pred = fx(x=x)
        _ = loss(predictions=y_pred, target=y)

    @pytest.mark.integration
    def test_simple_chain(self):
        data_source = nemo.backends.pytorch.tutorials.RealFunctionDataLayer(n=10000, batch_size=1)
        trainable_module = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        loss = nemo.backends.pytorch.tutorials.MSELoss()
        x, y = data_source()
        y_pred = trainable_module(x=x)
        loss_tensor = loss(predictions=y_pred, target=y)

        # check producers' bookkeeping
        self.assertEqual(loss_tensor.producer, loss)
        self.assertEqual(loss_tensor.producer_args, {"predictions": y_pred, "target": y})
        self.assertEqual(y_pred.producer, trainable_module)
        self.assertEqual(y_pred.producer_args, {"x": x})
        self.assertEqual(y.producer, data_source)
        self.assertEqual(y.producer_args, {})
        self.assertEqual(x.producer, data_source)
        self.assertEqual(x.producer_args, {})
