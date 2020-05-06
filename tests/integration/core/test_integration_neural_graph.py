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

from nemo.backends.pytorch.actions import PtActions
from nemo.backends.pytorch.tutorials import MSELoss, RealFunctionDataLayer, TaylorNet
from nemo.core import NeuralGraph


@pytest.mark.usefixtures("neural_factory")
class TestNeuralGraphTrainAction:
    @pytest.mark.integration
    def test_explicit_graph(self):
        """
            Tests the integration of an `explicit` graph with actions API.
            In particular, checks whether user can pass NG instance to train().
        """
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
            # Bind the loss output.
            g0.outputs["loss"] = lss

        # Instantiate an optimizer to perform the `train` action.
        optimizer = PtActions()

        # Make sure user CANNOT pass training graph and tensors_to_optimize.
        with pytest.raises(ValueError):
            optimizer.train(
                tensors_to_optimize=lss,
                training_graph=g0,
                optimization_params={"max_steps": 1, "lr": 0.0003},
                optimizer="sgd",
            )

        # But user can invoke "train" action using graph only.
        optimizer.train(training_graph=g0, optimization_params={"max_steps": 1, "lr": 0.0003}, optimizer="sgd")
