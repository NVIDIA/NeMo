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
import time

from nemo.backends.pytorch.tutorials import MSELoss, RealFunctionDataLayer, TaylorNet
from nemo.core import EvaluatorCallback, NeuralGraph, OperationMode

@pytest.mark.usefixtures("neural_factory")
class TestNeuralGraphSerialization:
    @pytest.mark.unit
    def test_graph_serialization_1_simple_graph_no_binding(self):
        """ 
            Tests whether serialization of a simple graph works.
        """
        # Instantiate the necessary neural modules.
        dl = RealFunctionDataLayer(n=100, batch_size=1, name="tgs1_dl")
        tn = TaylorNet(dim=4, name="tgs1_m1")
        loss = MSELoss(name="tgs1_loss")

        # Create the graph.
        with NeuralGraph(operation_mode=OperationMode.training, name="g1") as g1:
            x, t = dl()
            prediction1 = tn(x=x)
            lss = loss(predictions=prediction1, target=t)

        # Serialize the graph.
        serialized_g1 = g1.serialize()

        # Create a second graph - deserialize with reusing.
        g2 = NeuralGraph.deserialize(serialized_g1, reuse_existing_modules=True, name="g2")
        serialized_g2 = g2.serialize()

        # Must be the same.
        assert serialized_g1 == serialized_g2

        # Delete modules.
        del dl
        del tn
        del loss
        # Delete graphs as they contain "hard" references to those modules.
        del g1
        del g2

        # Create a third graph - deserialize without reusing, should create new modules.
        g3 = NeuralGraph.deserialize(serialized_g1, reuse_existing_modules=False, name="g3")
        serialized_g3 = g3.serialize()

        # Must be the same.
        assert serialized_g1 == serialized_g3

        # Deserialize graph - without reusing modules not allowed.
        with pytest.raises(KeyError):
            _ = NeuralGraph.deserialize(serialized_g1, reuse_existing_modules=False)


