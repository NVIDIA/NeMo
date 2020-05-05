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
from nemo.core import NeuralGraph, OperationMode
from nemo.core.neural_types import NeuralTypeComparisonResult
from nemo.utils.neural_graph.graph_outputs import GraphOutputs


@pytest.mark.usefixtures("neural_factory")
class TestGraphOutputs:
    @pytest.mark.unit
    def test_graph_outputs_binding1(self):
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
        with pytest.raises(TypeError):
            del bound_outputs["loss"]

        assert len(bound_outputs) == 4
        assert len(bound_outputs.tensors) == 4
        assert len(bound_outputs.tensor_list) == 4

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
        with pytest.raises(TypeError):
            del bound_outputs["my_prediction"]

        assert len(bound_outputs) == 2
        defs = bound_outputs.definitions
        assert defs["my_prediction"].compare(tn.output_ports["y_pred"]) == NeuralTypeComparisonResult.SAME
        assert defs["my_loss"].compare(loss.output_ports["loss"]) == NeuralTypeComparisonResult.SAME

        with pytest.raises(KeyError):
            _ = defs["x"]

    @pytest.mark.unit
    def test_graph_outputs_binding2(self):
        # Create modules.
        data_source = RealFunctionDataLayer(n=100, batch_size=1, name="tgo2_ds")
        tn = TaylorNet(dim=4, name="tgo2_tn")
        loss = MSELoss(name="tgo2_loss")

        # Test default binding.
        with NeuralGraph(operation_mode=OperationMode.training) as g1:
            # Create the graph by connnecting the modules.
            x, y = data_source()
            y_pred = tn(x=x)
            lss = loss(predictions=y_pred, target=y)

        assert len(g1.outputs) == 4
        # Test ports.
        for (module, port, tensor) in [
            (data_source, "x", x),
            (data_source, "y", y),
            (tn, "y_pred", y_pred),
            (loss, "loss", lss),
        ]:
            # Compare definitions - from outputs.
            assert g1.outputs[port].ntype.compare(module.output_ports[port]) == NeuralTypeComparisonResult.SAME
            # Compare definitions - from output_ports.
            assert g1.output_ports[port].compare(module.output_ports[port]) == NeuralTypeComparisonResult.SAME
            # Compare definitions - from output_tensors.
            assert g1.output_tensors[port].compare(module.output_ports[port]) == NeuralTypeComparisonResult.SAME
            # Make sure that tensor was bound, i.e. input refers to the same object instance!
            assert g1.output_tensors[port] is tensor

        # Test manual binding.
        g1.outputs["my_prediction"] = y_pred
        g1.outputs["my_loss"] = lss

        assert len(g1.outputs) == 2
        assert g1.output_tensors["my_prediction"].compare(tn.output_ports["y_pred"]) == NeuralTypeComparisonResult.SAME
        assert g1.output_tensors["my_loss"].compare(loss.output_ports["loss"]) == NeuralTypeComparisonResult.SAME

        # Finally, make sure that the user cannot "bind" "output_ports"!
        with pytest.raises(TypeError):
            g1.output_ports["my_prediction"] = y_pred

    @pytest.mark.unit
    def test_graph_inputs_binding1_default(self):
        # Create modules.
        tn = TaylorNet(dim=4, name="tgi1_tn")
        loss = MSELoss(name="tgi1_loss")

        # Test default binding.
        with NeuralGraph() as g1:
            y_pred = tn(x=g1)
            lss = loss(predictions=y_pred, target=g1)

        assert len(g1.inputs) == 2
        assert g1.input_ports["x"].compare(tn.input_ports["x"]) == NeuralTypeComparisonResult.SAME
        assert g1.input_ports["target"].compare(loss.input_ports["target"]) == NeuralTypeComparisonResult.SAME

    @pytest.mark.unit
    def test_graph_inputs_binding2_manual(self):
        # Create modules.
        tn = TaylorNet(dim=4, name="tgi2_tn")
        loss = MSELoss(name="tgi2_loss")

        # Test "manual" binding.
        with NeuralGraph() as g1:
            # Bind the "x" input to tn.
            g1.inputs["i"] = tn.input_ports["x"]
            y_pred = tn(x=g1.inputs["i"])
            # Bing the "target" input to loss.
            g1.inputs["t"] = loss.input_ports["target"]
            lss = loss(predictions=y_pred, target=g1.inputs["t"])

        assert len(g1.inputs) == 2
        assert g1.input_ports["i"].compare(tn.input_ports["x"]) == NeuralTypeComparisonResult.SAME
        assert g1.input_ports["t"].compare(loss.input_ports["target"]) == NeuralTypeComparisonResult.SAME

        # Finally, make sure that the user cannot "bind" "input_ports"!
        with pytest.raises(TypeError):
            g1.input_ports["my_prediction"] = y_pred
