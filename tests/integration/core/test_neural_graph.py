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

import pytest

from nemo.backends.pytorch.tutorials import MSELoss, RealFunctionDataLayer, TaylorNet
from nemo.core import NeuralGraph
from nemo.backends.pytorch.actions import PtActions

@pytest.mark.usefixtures("neural_factory")
class TestNeuralGraph:

    @pytest.mark.integration
    def test_nm_tensors(self):
        """
            Tests whether nmTensors are correct.
        """
        # Create modules.
        data_source = RealFunctionDataLayer(n=100, batch_size=1)
        trainable_module = TaylorNet(dim=4)
        loss = MSELoss()

        # Create the graph by connnecting the modules.
        x, y = data_source()
        y_pred = trainable_module(x=x)
        loss_tensor = loss(predictions=y_pred, target=y)

        # check producers' bookkeeping
        assert loss_tensor.producer == loss
        assert loss_tensor.producer_args == {"predictions": y_pred, "target": y}
        assert y_pred.producer == trainable_module
        assert y_pred.producer_args == {"x": x}
        assert y.producer == data_source
        assert y.producer_args == {}
        assert x.producer == data_source
        assert x.producer_args == {}

    @pytest.mark.integration
    def test_implicit_default_graph(self):
        """ Tests integration of a `default` (implicit) graph. """
        # Create modules.
        dl = RealFunctionDataLayer(n=100, batch_size=4)
        fx = TaylorNet(dim=4)
        loss = MSELoss()

        # This will create a default (implicit) graph: "training".
        x, t = dl()
        p = fx(x=x)
        lss = loss(predictions=p, target=t)

        # Instantiate an optimizer to perform the `train` action.
        optimizer = PtActions()
        # Invoke "train" action - perform single forward-backard step.
        optimizer.train([lss], optimization_params={"max_steps": 1, "lr": 0.0003}, optimizer="sgd")


    @pytest.mark.integration
    def test_explicit_graph(self):
        """  Tests integration of an `explicit` graph and decoupling of graph creation from its activation. """
        # Create modules.
        dl = RealFunctionDataLayer(n=100, batch_size=4)
        fx = TaylorNet(dim=4)
        loss = MSELoss()

        # Create the g0 graph.
        g0 = NeuralGraph()

        # Activate the "g0 graph context" - all operations will be recorded to g0.
        with g0:
            x, t = dl()
            p = fx(x=x)
            lss = loss(predictions=p, target=t)

        # Instantiate an optimizer to perform the `train` action.
        optimizer = PtActions()
        # Invoke "train" action - perform single forward-backard step.
        optimizer.train([lss], optimization_params={"max_steps": 1, "lr": 0.0003}, optimizer="sgd")

