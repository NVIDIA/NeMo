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
from nemo.core.neural_graph.graph_outputs import GraphOutputs
from nemo.core.neural_types import NeuralTypeComparisonResult


@pytest.mark.usefixtures("neural_factory")
class TestGraphOutputs:
    @pytest.mark.unit
    def test_binding(self):
        # Create modules.
        data_source = RealFunctionDataLayer(n=100, batch_size=1)
        tn = TaylorNet(dim=4)
        loss = MSELoss()

        with NeuralGraph() as g:
            # Create the graph by connnecting the modules.
            x, y = data_source()
            y_pred = tn(x=x)
            lss = loss(predictions=y_pred, target=y)

        # Test default binding.
        bound_outputs = GraphOutputs(g.tensors)

        bound_outputs.bind([x, y])
        bound_outputs.bind([y_pred])
        bound_outputs.bind([lss])

        # Delete not allowed.
        with pytest.raises(NotImplementedError):
            del bound_outputs["loss"]

        assert len(bound_outputs) == 4

        defs = bound_outputs.definitions
        assert defs["x"].compare(data_source.output_ports["x"]) == NeuralTypeComparisonResult.SAME
        assert defs["y"].compare(data_source.output_ports["y"]) == NeuralTypeComparisonResult.SAME
        assert defs["y_pred"].compare(tn.output_ports["y_pred"]) == NeuralTypeComparisonResult.SAME
        assert defs["loss"].compare(loss.output_ports["loss"]) == NeuralTypeComparisonResult.SAME

        with pytest.raises(KeyError):
            _ = defs["lss"]

        # Bound manually.
        bound_outputs["my_prediction"] = y_pred
        bound_outputs["my_loss"] = lss

        # Delete not allowed.
        with pytest.raises(NotImplementedError):
            del bound_outputs["my_prediction"]

        assert len(bound_outputs) == 2
        defs = bound_outputs.definitions
        assert defs["my_prediction"].compare(tn.output_ports["y_pred"]) == NeuralTypeComparisonResult.SAME
        assert defs["my_loss"].compare(loss.output_ports["loss"]) == NeuralTypeComparisonResult.SAME

        with pytest.raises(KeyError):
            _ = defs["x"]
