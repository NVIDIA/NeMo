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


import pytest
from numpy import array_equal

from nemo.backends import get_state_dict
from nemo.backends.pytorch.tutorials import MSELoss, RealFunctionDataLayer, TaylorNet
from nemo.core import NeuralGraph
from nemo.core.neural_types import NeuralTypeComparisonResult


@pytest.mark.usefixtures("neural_factory")
class TestNeuralGraphs:
    @pytest.mark.unit
    def test_explicit_graph_with_activation(self):
        """ 
            Tests initialization of an `explicit` graph and decoupling of graph creation from its activation. 
            Also tests modules access.
        """
        # Create modules.
        dl = RealFunctionDataLayer(n=10, batch_size=1, name="dl")
        fx = TaylorNet(dim=4, name="fx")
        loss = MSELoss(name="loss")

        # Create the g0 graph.
        g0 = NeuralGraph()

        # Activate the "g0 graph context" - all operations will be recorded to g0.
        with g0:
            x, t = dl()
            p = fx(x=x)
            lss = loss(predictions=p, target=t)

        # Assert that there are 3 modules in the graph.
        assert len(g0) == 3

        # Test access modules.
        assert g0["dl"] is dl
        assert g0["fx"] is fx
        assert g0["loss"] is loss

        with pytest.raises(KeyError):
            g0["other_module"]

    @pytest.mark.unit
    def test_explicit_graph_manual_activation(self):
        """  Tests initialization of an `explicit` graph using `manual` activation. """
        # Create modules.
        dl = RealFunctionDataLayer(n=10, batch_size=1)
        fx = TaylorNet(dim=4)

        # Create the g0 graph.
        g0 = NeuralGraph()

        # Activate the "g0 graph context" "manually" - all steps will be recorded to g0.
        g0.activate()

        # Define g0 - connections between the modules.
        x, t = dl()
        p = fx(x=x)

        # Deactivate the "g0 graph context".
        # Note that this is really optional, as long as there are no other steps to be recorded.
        g0.deactivate()

        # Assert that there are 2 modules in the graph.
        assert len(g0) == 2

    @pytest.mark.unit
    def test_graph_save_load(self, tmpdir):
        """
            Tests graph saving and loading.
        
            Args:
                tmpdir: Fixture which will provide a temporary directory.
        """

        dl = RealFunctionDataLayer(n=10, batch_size=1)
        tn = TaylorNet(dim=4)
        # Get the "original" weights.
        weights1 = get_state_dict(tn)

        # Create a simple graph.
        with NeuralGraph() as g1:
            x, t = dl()
            p = tn(x=x)

        # Generate filename in the temporary directory.
        tmp_file_name = str(tmpdir.join("tgsl_g1.chkpt"))
        # Save graph.
        g1.save_to(tmp_file_name)

        # Load graph.
        g1.restore_from(tmp_file_name)

        # Get the "restored" weights.
        weights2 = get_state_dict(tn)

        # Compare state dicts.
        for key in weights1:
            assert array_equal(weights1[key].cpu().numpy(), weights2[key].cpu().numpy())
