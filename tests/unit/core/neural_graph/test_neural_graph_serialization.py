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


@pytest.mark.usefixtures("neural_factory")
class TestNeuralGraphSerialization:
    @pytest.mark.unit
    def test_graph_serialization_1_simple_graph_no_binding(self):
        """ 
            Tests whether serialization of a simple graph works.
        """
        # Instantiate the necessary neural modules.
        dl = RealFunctionDataLayer(n=100, batch_size=1, name="tgs1_dl")
        tn = TaylorNet(dim=4, name="tgs1_tn")
        loss = MSELoss(name="tgs1_loss")

        # Create the graph.
        with NeuralGraph(operation_mode=OperationMode.training, name="g1") as g1:
            x, t = dl()
            prediction1 = tn(x=x)
            _ = loss(predictions=prediction1, target=t)

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

    @pytest.mark.unit
    def test_graph_serialization_2_simple_graph_output_binding(self):
        """ 
            Tests whether serialization of a simple graph with output binding works.
        """
        # Instantiate the necessary neural modules.
        dl = RealFunctionDataLayer(n=100, batch_size=1, name="tgs2_dl")
        tn = TaylorNet(dim=4, name="tgs2_tn")
        loss = MSELoss(name="tgs2_loss")

        # Create the graph.
        with NeuralGraph(operation_mode=OperationMode.evaluation) as g1:
            x, t = dl()
            prediction1 = tn(x=x)
            _ = loss(predictions=prediction1, target=t)
        # Manually bind the selected outputs.
        g1.outputs["ix"] = x
        g1.outputs["te"] = t
        g1.outputs["prediction"] = prediction1

        # Serialize graph
        serialized_g1 = g1.serialize()

        # Create the second graph - deserialize with reusing.
        g2 = NeuralGraph.deserialize(serialized_g1, reuse_existing_modules=True)
        serialized_g2 = g2.serialize()

        # Must be the same.
        assert serialized_g1 == serialized_g2

    @pytest.mark.unit
    def test_graph_serialization_3_simple_model_input_output_binding(self):
        """ 
            Tests whether serialization of a simple graph with input and output binding works.
        """
        # Instantiate the necessary neural modules.
        tn = TaylorNet(dim=4, name="tgs3_tn")

        # Create "model".
        with NeuralGraph(operation_mode=OperationMode.both, name="model") as model:
            # Manually bind input port: "input" -> "x"
            model.inputs["input"] = tn.input_ports["x"]
            # Add module to graph and bind it input port 'x'.
            y = tn(x=model.inputs["input"])
            # Manual output bind.
            model.outputs["output"] = y

        # Serialize the "model".
        serialized_model1 = model.serialize()

        # Create the second graph - deserialize with reusing.
        model2 = NeuralGraph.deserialize(serialized_model1, reuse_existing_modules=True)
        serialized_model2 = model2.serialize()

        # Must be the same.
        assert serialized_model1 == serialized_model2

    @pytest.mark.unit
    def test_graph_serialization_4_graph_after_nesting_with_default_binding_reuse_modules(self):
        """ 
            Tests whether serialization works in the case when we serialize a graph after a different graph
            was nested in it, with additionally bound input and output binding works (default port names).
        """
        # Instantiate the necessary neural modules.
        dl = RealFunctionDataLayer(n=100, batch_size=1, name="tgs4_dl")
        tn = TaylorNet(dim=4, name="tgs4_tn")
        loss = MSELoss(name="tgs4_loss")

        # Create "model".
        with NeuralGraph(operation_mode=OperationMode.both, name="model") as model:
            # Add module to graph and bind it input port 'x'.
            y = tn(x=model)
            # NOTE: For some reason after this call both the "tgs4_tn" and "model" objects
            # remains on the module/graph registries.
            # (So somewhere down there remains a strong reference to module or graph).
            # This happens ONLY when passing graph as argument!
            # (Check out the next test which actually removes module and graph!).
            # Still, that is not an issue, as we do not expect the users
            # to delete and recreate modules in their "normal" applications.

        # Build the "training graph" - using the model copy.
        with NeuralGraph(operation_mode=OperationMode.training, name="tgs4_training") as training:
            # Add modules to graph.
            x, t = dl()
            # Incorporate modules from the existing "model" graph.
            p = model(x=x)
            lss = loss(predictions=p, target=t)

        # Serialize the "training graph".
        serialized_training = training.serialize()

        # Create the second graph - deserialize withoput "module reusing".
        training2 = NeuralGraph.deserialize(serialized_training, reuse_existing_modules=True)
        serialized_training2 = training2.serialize()

        # Must be the same.
        assert serialized_training == serialized_training2

    @pytest.mark.unit
    def test_graph_serialization_5_graph_after_nesting_without_reusing(self):
        """ 
            Tests whether serialization works in the case when we serialize a graph after a different graph
            was nested in it, with additionally bound input and output binding works (default port names).
        """
        # Instantiate the necessary neural modules.
        dl = RealFunctionDataLayer(n=100, batch_size=1, name="tgs5_dl")
        tn = TaylorNet(dim=4, name="tgs511_tn")
        loss = MSELoss(name="tgs5_loss")

        # Create "model".
        with NeuralGraph(operation_mode=OperationMode.both, name="tgs5_model") as model:
            # Manually bind input port: "input" -> "x"
            model.inputs["input"] = tn.input_ports["x"]
            # Add module to graph and bind it input port 'x'.
            y = tn(x=model.inputs["input"])
            # Use the default output name.

        # Build the "training graph" - using the model copy.
        with NeuralGraph(operation_mode=OperationMode.training, name="tgs5_training") as training:
            # Add modules to graph.
            x, t = dl()
            # Incorporate modules from the existing "model" graph.
            p = model(input=x)
            lss = loss(predictions=p, target=t)

        # Serialize the "training graph".
        serialized_training = training.serialize()

        # Delete everything.
        del dl
        del tn
        del loss
        del model
        del training

        # Create the second graph - deserialize withoput "module reusing".
        training2 = NeuralGraph.deserialize(serialized_training)
        serialized_training2 = training2.serialize()

        # Must be the same.
        assert serialized_training == serialized_training2

    @pytest.mark.unit
    def test_graph_serialization_6_graph_after_nesting_with_manual_binding(self):
        """ 
            Tests whether serialization works in the case when we serialize a graph after a different graph
            was nested in it, with additionally bound input and output binding works (manual port names).
        """
        # Instantiate the necessary neural modules.
        dl = RealFunctionDataLayer(n=100, batch_size=1, name="tgs6_dl")
        tn = TaylorNet(dim=4, name="tgs6_tn")
        loss = MSELoss(name="tgs6_loss")

        # Create "model".
        with NeuralGraph(operation_mode=OperationMode.both, name="tgs6_model") as model:
            # Manually bind input port: "input" -> "x"
            model.inputs["input"] = tn.input_ports["x"]
            # Add module to graph and bind it input port 'x'.
            y = tn(x=model.inputs["input"])
            # Manual output bind.
            model.outputs["output"] = y

        # Serialize "model".
        serialized_model = model.serialize()

        # Delete model-related stuff.
        del model
        del tn

        # Deserialize the "model copy".
        model_copy = NeuralGraph.deserialize(serialized_model, name="tgs6_model_copy")

        # Build the "training graph" - using the model copy.
        with NeuralGraph(operation_mode=OperationMode.training, name="tgs6_training") as training:
            # Add modules to graph.
            x, t = dl()
            # Incorporate modules from the existing "model" graph.
            p = model_copy(input=x)  # Note: this output should actually be named "output", not "y_pred"!
            lss = loss(predictions=p, target=t)

        # Serialize the "training graph".
        serialized_training = training.serialize()

        # Delete everything.
        del dl
        del loss
        del model_copy
        del training

        # Create the second graph - deserialize without "module reusing".
        training2 = NeuralGraph.deserialize(serialized_training)
        serialized_training2 = training2.serialize()

        # Must be the same.
        assert serialized_training == serialized_training2

    @pytest.mark.unit
    def test_graph_serialization_7_arbitrary_graph_with_loops(self):
        """ 
            Tests whether serialization works in the case when we serialize a graph after a different graph
            was nested in it, with additionally bound input and output binding works (manual port names).
        """
        # Instantiate the necessary neural modules.
        dl = RealFunctionDataLayer(n=100, batch_size=1, name="dl")
        tn = TaylorNet(dim=4, name="tn")
        loss = MSELoss(name="loss")

        # Build a graph with a loop.
        with NeuralGraph(name="graph") as graph:
            # Add modules to graph.
            x, t = dl()
            # First call to TN.
            p1 = tn(x=x)
            # Second call to TN.
            p2 = tn(x=p1)
            # Take output of second, pass it to loss.
            lss = loss(predictions=p2, target=t)

        # Make sure all connections are there!
        assert len(graph.tensor_list) == 5
        # 4 would mean that we have overwritten the "p1" (tn->y_pred) tensor!

        # Serialize the graph.
        serialized_graph = graph.serialize()

        # Create the second graph - deserialize with "module reusing".
        graph2 = NeuralGraph.deserialize(serialized_graph, reuse_existing_modules=True)
        serialized_graph2 = graph2.serialize()

        # Must be the same.
        assert serialized_graph == serialized_graph2

        # import pdb;pdb.set_trace()
        # print("1: \n",serialized_graph)
        # print("2: \n",serialized_graph2)
