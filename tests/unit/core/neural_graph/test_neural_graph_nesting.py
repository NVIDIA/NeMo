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
import torch

from nemo.backends.pytorch.actions import PtActions
from nemo.backends.pytorch.tutorials import MSELoss, RealFunctionDataLayer, TaylorNet
from nemo.core import EvaluatorCallback, NeuralGraph, OperationMode, SimpleLossLoggerCallback
from nemo.core.neural_types import NeuralTypeComparisonResult
from nemo.utils import logging


@pytest.mark.usefixtures("neural_factory")
class TestNeuralGraphNesting:
    @pytest.mark.unit
    def test_module_nesting1_change_operation_modes(self):
        """ 
            Tests whether invalid nesting (i.e. nesting of graphs with incompatible modes) throw exeptions.
        """
        # Instantiate the necessary neural modules.
        dl = RealFunctionDataLayer(n=10, batch_size=1)

        with NeuralGraph(operation_mode=OperationMode.both):
            _, _ = dl()
            assert dl.operation_mode == OperationMode.both

        with NeuralGraph(operation_mode=OperationMode.training):
            _, _ = dl()
            assert dl.operation_mode == OperationMode.training

        with NeuralGraph(operation_mode=OperationMode.evaluation):
            _, _ = dl()
            assert dl.operation_mode == OperationMode.evaluation

    @pytest.mark.unit
    def test_graph_nesting2_possible_operation_modes(self):
        """ 
            Tests whether invalid nesting (i.e. nesting of graphs with incompatible modes) throw exeptions.
        """
        # Instantiate the necessary neural modules.
        dl = RealFunctionDataLayer(n=10, batch_size=1)

        with NeuralGraph(operation_mode=OperationMode.both) as both:
            _, _ = dl()

        with NeuralGraph(operation_mode=OperationMode.training) as training:
            _, _ = dl()

        with NeuralGraph(operation_mode=OperationMode.evaluation) as inference:
            _, _ = dl()

        # Allowed operations.
        # Can nest 'both' into 'training'.
        with NeuralGraph(operation_mode=OperationMode.training):
            _, _ = both()

        # Can nest 'both' into 'inference'.
        with NeuralGraph(operation_mode=OperationMode.evaluation):
            _, _ = both()

        # Can nest 'training' into 'training'.
        with NeuralGraph(operation_mode=OperationMode.training):
            _, _ = training()

        # Can nest 'inference' into 'inference'.
        with NeuralGraph(operation_mode=OperationMode.evaluation):
            _, _ = inference()

        # Can nest 'both' into 'both'.
        with NeuralGraph(operation_mode=OperationMode.both):
            _, _ = both()

        # Operations not allowed.
        # Cannot nest 'inference' into 'training'.
        with pytest.raises(TypeError):
            with NeuralGraph(operation_mode=OperationMode.training):
                _, _ = inference()

        # Cannot nest 'training' into 'inference'.
        with pytest.raises(TypeError):
            with NeuralGraph(operation_mode=OperationMode.evaluation):
                _, _ = training()

        # Cannot nest 'training' into 'both'.
        with pytest.raises(TypeError):
            with NeuralGraph(operation_mode=OperationMode.both):
                _, _ = training()

        # Cannot nest 'inference' into 'both'.
        with pytest.raises(TypeError):
            with NeuralGraph(operation_mode=OperationMode.both):
                _, _ = inference()

    @pytest.mark.unit
    def test_graph_nesting3_topology_copy_one_module_default_outputs(self):
        """
            Test whether when nesting of one graph into another will result in copy of the graph topology (tensors).
            Case: binding of outputs, default port names.
        """
        dl = RealFunctionDataLayer(n=10, batch_size=1, name="tgn3_dl")

        # Create the "inner graph".
        with NeuralGraph(operation_mode=OperationMode.training, name="tgn3_g1") as g1:
            xg1, tg1 = dl()

        # Create the "outer graph".
        with NeuralGraph(operation_mode=OperationMode.training, name="tgn3_g2") as g2:
            xg2, tg2 = g1()

        # We expect that both graphs will have the same steps.
        assert len(g1.steps) == len(g2.steps)
        assert g1.steps[0] == g2.steps[0]

        # Make sure that the modules are the same.
        assert len(g1) == len(g2)
        assert g1["tgn3_dl"] is dl
        assert g2["tgn3_dl"] is dl
        assert g1["tgn3_dl"] is g2["tgn3_dl"]

        # Make sure that outputs are ok.
        assert len(g1.outputs) == len(g2.outputs)
        for port in ["x", "y"]:
            # Definitions are the same: test two "paths" of accessing the type.
            assert g1.outputs[port].ntype.compare(g1.output_ports[port]) == NeuralTypeComparisonResult.SAME

            assert g1.output_ports[port].compare(g2.output_ports[port]) == NeuralTypeComparisonResult.SAME
            assert g1.outputs[port].ntype.compare(g2.outputs[port].ntype) == NeuralTypeComparisonResult.SAME
            # At the same time - those have to be two different port objects!
            assert g1.outputs[port] is not g2.outputs[port]
            # And different tensors (as those are "internally produced tensors"!)
            assert g1.output_tensors[port] is not g2.output_tensors[port]

    @pytest.mark.unit
    def test_graph_nesting4_topology_copy_one_module_manual_outputs(self):
        """
            Test whether when nesting of one graph into another will result in copy of the graph topology (tensors).
            Case: binding of outputs, manual port names.
        """

        dl = RealFunctionDataLayer(n=10, batch_size=1, name="tgn4_dl")

        # Create the "inner graph".
        with NeuralGraph(operation_mode=OperationMode.training, name="tgn4_g1") as g1:
            xg1, tg1 = dl()
            # Set port binding manually, with different names - and their number!
            g1.outputs["inner_x"] = xg1

        # Create the "outer graph".
        with NeuralGraph(operation_mode=OperationMode.training, name="tgn4_g2") as g2:
            xg2 = g1()
            # Set port binding manually, with different names - and their number!
            g2.outputs["outer_x"] = xg2

        # We expect that both graphs will have the same steps.
        assert len(g1.steps) == len(g2.steps)
        assert g1.steps[0] == g2.steps[0]

        # Make sure that the modules are the same.
        assert len(g1) == len(g2)
        assert g1["tgn4_dl"] is g2["tgn4_dl"]

        # Make sure that outputs are ok.
        assert len(g1.outputs) == len(g2.outputs)
        for inter_port, outer_port in [("inner_x", "outer_x")]:
            # Definitions are the same: test two "paths" of accessing the type.
            assert g1.output_ports[inter_port].compare(g2.output_ports[outer_port]) == NeuralTypeComparisonResult.SAME
            assert (
                g1.outputs[inter_port].ntype.compare(g2.outputs[outer_port].ntype) == NeuralTypeComparisonResult.SAME
            )
            # At the same time - those have to be two different port objects!
            assert g1.outputs[inter_port] is not g2.outputs[outer_port]
            # And different tensors (as those are "internally produced tensors"!)
            assert g1.output_tensors[inter_port] is not g2.output_tensors[outer_port]

    @pytest.mark.unit
    def test_graph_nesting4_1_topology_copy_one_module_manual_outputs_bound_only_in_inner(self):
        """
            Test whether when nesting of one graph into another will result in copy of the graph topology (tensors).
            Case: binding of outputs, manual port names - only in the inner graph.
            Testing whether outputs of outer graph have the manually bound names.
        """

        dl = RealFunctionDataLayer(n=10, batch_size=1, name="tgn41_dl")

        # Create the "inner graph".
        with NeuralGraph(operation_mode=OperationMode.training, name="tgn41_g1") as g1:
            xg1, tg1 = dl()
            # Set port binding manually, with different names - and their number!
            g1.outputs["inner_x"] = xg1
            g1.outputs["inner_t"] = tg1

        # Create the "outer graph".
        with NeuralGraph(operation_mode=OperationMode.training, name="tgn41_g2") as g2:
            # Get them as a tuple.
            outputs = g1()

        # Retrieve tensors from tuple.
        assert outputs._fields[0] == "inner_x"
        assert outputs._fields[1] == "inner_t"
        xg2 = outputs.inner_x
        tg2 = outputs.inner_t

        # Make sure that outer graph has objects of the same names
        assert len(g1.outputs) == len(g2.outputs)
        for inter_port, outer_port in [("inner_x", "inner_x"), ("inner_t", "inner_t")]:
            # Definitions are the same: test two "paths" of accessing the type.
            assert g1.output_ports[inter_port].compare(g2.output_ports[outer_port]) == NeuralTypeComparisonResult.SAME
            assert (
                g1.outputs[inter_port].ntype.compare(g2.outputs[outer_port].ntype) == NeuralTypeComparisonResult.SAME
            )
            # At the same time - those have to be two different port objects!
            assert g1.outputs[inter_port] is not g2.outputs[outer_port]
            # And different tensors (as those are "internally produced tensors"!)
            assert g1.output_tensors[inter_port] is not g2.output_tensors[outer_port]

    @pytest.mark.unit
    def test_graph_nesting5_topology_copy_one_module_default_inputs(self):
        """
            Test whether when nesting of one graph into another will result in copy of the graph topology (tensors).
            Case: binding of inputs, default port names.
        """
        tn = TaylorNet(dim=4, name="tgn5_tn")

        # Create the "inner graph".
        with NeuralGraph(operation_mode=OperationMode.training) as g1:
            y_pred1 = tn(x=g1)

        # Create the "outer graph".
        with NeuralGraph(operation_mode=OperationMode.training) as g2:
            y_pred2 = g1(x=g2)

        # We expect that both graphs will have the same steps.
        assert len(g1.steps) == len(g2.steps)
        assert g1.steps[0] == g2.steps[0]

        # Make sure that the modules are the same.
        assert len(g1) == len(g2)
        assert g1["tgn5_tn"] is g2["tgn5_tn"]

        # Make sure that inputs are ok.
        assert len(g1.inputs) == len(g2.inputs)
        assert g1.input_ports["x"].compare(tn.input_ports["x"]) == NeuralTypeComparisonResult.SAME
        assert g2.input_ports["x"].compare(tn.input_ports["x"]) == NeuralTypeComparisonResult.SAME
        # At the same time - those point to the same step-module-port.
        assert g1.inputs.has_binding(0, "x")
        assert g2.inputs.has_binding(0, "x")
        assert g1.inputs["x"].consumers[0].step_number == 0
        assert g1.inputs["x"].consumers[0].module_name == tn.name
        assert g1.inputs["x"].consumers[0].port_name == "x"
        assert g2.inputs["x"].consumers[0].step_number == 0
        assert g2.inputs["x"].consumers[0].module_name == tn.name
        assert g2.inputs["x"].consumers[0].port_name == "x"

        # Make sure that outputs are ok.
        assert len(g1.outputs) == len(g2.outputs)
        assert g1.output_ports["y_pred"].compare(tn.output_ports["y_pred"]) == NeuralTypeComparisonResult.SAME
        assert g1.output_ports["y_pred"].compare(tn.output_ports["y_pred"]) == NeuralTypeComparisonResult.SAME
        # At the same time - those have to be two different port objects!
        assert g1.outputs["y_pred"] is not g2.outputs["y_pred"]
        # And different tensors (as those are "internally produced tensors"!)
        assert g1.output_tensors["y_pred"] is y_pred1
        assert g2.output_tensors["y_pred"] is y_pred2
        assert y_pred1 is not y_pred2

    @pytest.mark.unit
    def test_graph_nesting6_topology_copy_one_module_manual_inputs(self):
        """
            Test whether when nesting of one graph into another will result in copy of the graph topology (tensors).
            Case: binding of inputs, manual port names.
        """
        tn = TaylorNet(dim=4, name="tgn6_tn")

        # Create the "inner graph".
        with NeuralGraph(operation_mode=OperationMode.training, name="tgn6_g1") as g1:
            # Copy input type.
            g1.inputs["inner_x"] = tn.input_ports["x"]
            # Bind the input port.
            y_pred1 = tn(x=g1.inputs["inner_x"])

        # Create the "outer graph".
        with NeuralGraph(operation_mode=OperationMode.training, name="tgn6_g2") as g2:
            # Copy input type.
            g2.inputs["outer_x"] = g1.input_ports["inner_x"]
            # Bind the input port.
            y_pred2 = g1(inner_x=g2.inputs["outer_x"])

        # We expect that both graphs will have the same steps.
        assert len(g1.steps) == len(g2.steps)
        assert g1.steps[0] == g2.steps[0]

        # Make sure that the modules are the same.
        assert len(g1) == len(g2)
        assert g1["tgn6_tn"] is g2["tgn6_tn"]

        # Make sure that inputs are ok.
        assert len(g1.inputs) == len(g2.inputs)
        assert g1.input_ports["inner_x"].compare(tn.input_ports["x"]) == NeuralTypeComparisonResult.SAME
        assert g2.input_ports["outer_x"].compare(tn.input_ports["x"]) == NeuralTypeComparisonResult.SAME
        # At the same time - those point to the same module-port.
        assert g1.inputs.has_binding(0, "x")
        assert g2.inputs.has_binding(0, "x")
        assert g1.inputs["inner_x"].consumers[0].step_number == 0
        assert g1.inputs["inner_x"].consumers[0].module_name == tn.name
        assert g1.inputs["inner_x"].consumers[0].port_name == "x"
        assert g2.inputs["outer_x"].consumers[0].step_number == 0
        assert g2.inputs["outer_x"].consumers[0].module_name == tn.name
        assert g2.inputs["outer_x"].consumers[0].port_name == "x"

        # Make sure that outputs are ok.
        assert len(g1.outputs) == len(g2.outputs)
        assert g1.output_ports["y_pred"].compare(tn.output_ports["y_pred"]) == NeuralTypeComparisonResult.SAME
        assert g1.output_ports["y_pred"].compare(tn.output_ports["y_pred"]) == NeuralTypeComparisonResult.SAME
        # At the same time - those have to be two different port objects!
        assert g1.outputs["y_pred"] is not g2.outputs["y_pred"]
        # And different tensors (as those are "internally produced tensors"!)
        assert g1.output_tensors["y_pred"] is y_pred1
        assert g2.output_tensors["y_pred"] is y_pred2
        assert y_pred1 is not y_pred2

    @pytest.mark.unit
    def test_graph_nesting7_topology_copy_one_module_all_manual_connect(self):
        """
            Test whether when nesting of one graph into another will result in copy of the graph topology (tensors).
            Case: manual binding of inputs and outputs, connects to other modules.
        """
        ds = RealFunctionDataLayer(n=10, batch_size=1, name="tgn7_ds")
        tn = TaylorNet(dim=4, name="tgn7_tn")
        loss = MSELoss(name="tgn7_loss")

        # Create the "inner graph".
        with NeuralGraph(operation_mode=OperationMode.training, name="tgn7_g1") as g1:
            # Copy the input type.
            g1.inputs["inner_x"] = tn.input_ports["x"]
            # Manually bind the input port.
            y_pred1 = tn(x=g1.inputs["inner_x"])
            # Manually bind the output port.
            g1.outputs["inner_y_pred"] = y_pred1

        # Create the "outer graph".
        with NeuralGraph(operation_mode=OperationMode.training, name="tgn7_g2") as g2:
            x, y = ds()
            y_pred2 = g1(inner_x=x)
            lss = loss(predictions=y_pred2, target=y)

        # Check steps.
        assert len(g2.steps) == 3
        assert g2.steps[1] == g1.steps[0]

        # Make sure that the modules are the same.
        assert len(g2) == 3
        assert g2["tgn7_tn"] is g1["tgn7_tn"]

        # Make sure that inputs are ok.
        assert len(g2.inputs) == 0

        # Check outputs.
        assert len(g2.outputs) == 4
        assert g2.output_ports["x"].compare(ds.output_ports["x"]) == NeuralTypeComparisonResult.SAME
        assert g2.output_ports["y"].compare(ds.output_ports["y"]) == NeuralTypeComparisonResult.SAME
        assert g2.output_ports["loss"].compare(loss.output_ports["loss"]) == NeuralTypeComparisonResult.SAME
        # The manually bound name!
        assert g2.output_ports["inner_y_pred"].compare(tn.output_ports["y_pred"]) == NeuralTypeComparisonResult.SAME

        # Check the output tensors.
        assert len(g2.output_tensors) == 4
        assert g2.output_tensors["x"] == x
        assert g2.output_tensors["y"] == y
        assert g2.output_tensors["loss"] == lss
        # The manually bound name!
        assert g2.output_tensors["inner_y_pred"] == y_pred2

        # Check the "internal tensors".
        assert y_pred2 is not y_pred1
        assert g2.tensors[0]["x"] == x
        assert g2.tensors[0]["y"] == y
        assert g2.tensors[2]["loss"] == lss
        # Internally the name "y_pred" is used, not the "bound output name": "inner_y_pred"!
        assert g2.tensors[1]["y_pred"] == y_pred2

        # Update g2: manually bound only one output.
        with g2:
            g2.outputs["outer_loss"] = lss

        # Make sure that outputs are ok.
        assert len(g2.outputs) == 1
        assert g2.output_ports["outer_loss"].compare(loss.output_ports["loss"]) == NeuralTypeComparisonResult.SAME
        assert g2.output_tensors["outer_loss"] is lss

    @pytest.mark.unit
    def test_graph_nesting8_topology_copy_two_modules(self):
        """
            Test whether when nesting of one graph into another will result in copy of the graph topology (tensors).
            Case: manual binding of inputs and outputs in the inner graph.
        """
        ds = RealFunctionDataLayer(n=10, batch_size=1, name="tgn8_ds")
        tn = TaylorNet(dim=4, name="tgn8_tn")
        loss = MSELoss(name="tgn8_loss")

        # Create the "inner graph".
        with NeuralGraph(operation_mode=OperationMode.training, name="tgn8_g1") as g1:
            # Create input port definitions.
            g1.inputs["inner_x"] = tn.input_ports["x"]
            g1.inputs["inner_target"] = loss.input_ports["target"]

            # Connect modules and bound inputs.
            y_pred1 = tn(x=g1.inputs["inner_x"])
            lss1 = loss(predictions=y_pred1, target=g1.inputs["inner_target"])

            # Manually bind the output ports.
            g1.outputs["inner_y_pred"] = y_pred1
            g1.outputs["inner_loss"] = lss1

        # Create the "outer graph".
        with NeuralGraph(operation_mode=OperationMode.training, name="tgn8_g2") as g2:
            x, y = ds()
            # Nest the inner graph.
            y_pred2, lss2 = g1(inner_x=x, inner_target=y)
            # Manually bind the output ports.
            g2.outputs["outer_y_pred"] = y_pred2
            g2.outputs["outer_loss"] = lss2

        # Check modules and steps.
        assert len(g2.steps) == 3
        assert len(g2) == 3

        # Check the output tensors.
        assert len(g2.output_tensors) == 2
        assert g2.output_tensors["outer_y_pred"] == y_pred2
        assert g2.output_tensors["outer_loss"] == lss2

        # Check the "internal tensors".
        assert y_pred2 is not y_pred1
        assert lss2 is not lss1
        assert g2.tensors[0]["x"] == x
        assert g2.tensors[0]["y"] == y
        # Internally the name "y_pred" is used, not the "bound output name": "inner_y_pred"!
        assert g2.tensors[1]["y_pred"] == y_pred2
        # Analogically with "loss".
        assert g2.tensors[2]["loss"] == lss2

    @pytest.mark.unit
    def test_graph_nesting9_topology_copy_whole_graph(self):
        """
            Test whether when nesting of one graph into another will result in copy of the graph topology (tensors).
            Case: manual binding of inputs and outputs in the inner graph. Manual binding of outer graph outputs.
        """
        ds = RealFunctionDataLayer(n=10, batch_size=1, name="tgn9_ds")
        tn = TaylorNet(dim=4, name="tgn9_tn")
        loss = MSELoss(name="tgn9_loss")

        # Create the "inner graph".
        with NeuralGraph(operation_mode=OperationMode.training, name="tgn9_g1") as g1:
            # Connect modules.
            x, y = ds()
            y_pred1 = tn(x=x)
            lss1 = loss(predictions=y_pred1, target=y)

            # Manually bind the output ports.
            g1.outputs["inner_y_pred"] = y_pred1
            g1.outputs["inner_loss"] = lss1

        # Create the "outer graph".
        with NeuralGraph(operation_mode=OperationMode.training, name="tgn9_g2") as g2:
            y_pred2, lss2 = g1()
            # Manually bind the output ports.
            g2.outputs["outer_y_pred"] = y_pred2
            g2.outputs["outer_loss"] = lss2

        # Check modules and steps.
        assert len(g2.steps) == 3
        assert len(g2) == 3

        # Check the output tensors.
        assert len(g2.output_tensors) == 2
        assert g2.output_tensors["outer_y_pred"] == y_pred2
        assert g2.output_tensors["outer_loss"] == lss2

        # Check the "internal tensors".
        assert y_pred2 is not y_pred1
        assert lss2 is not lss1
        assert g2.tensors[0]["x"].ntype.compare(ds.output_ports["x"]) == NeuralTypeComparisonResult.SAME
        assert g2.tensors[0]["y"].ntype.compare(ds.output_ports["y"]) == NeuralTypeComparisonResult.SAME
        # Internally the name "y_pred" is used, not the "bound output name": "inner_y_pred"!
        assert g2.tensors[1]["y_pred"].ntype.compare(tn.output_ports["y_pred"]) == NeuralTypeComparisonResult.SAME
        # Analogically with "loss".
        assert g2.tensors[2]["loss"].ntype.compare(loss.output_ports["loss"]) == NeuralTypeComparisonResult.SAME
