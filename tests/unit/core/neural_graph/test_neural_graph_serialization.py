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

import time

import pytest

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
        with NeuralGraph(operation_mode=OperationMode.inference) as g1:
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
    def test_graph_serialization_4_serialize_graph_after_nesting(self):
        """ 
            Tests whether serialization of a simple graph with input and output binding works.
        """
        # Instantiate the necessary neural modules.
        dl = RealFunctionDataLayer(n=100, batch_size=1)
        tn = TaylorNet(dim=4)
        loss = MSELoss()

        # Create "model".
        with NeuralGraph(operation_mode=OperationMode.both, name="model") as model:
            # Manually bind input port: "input" -> "x"
            model.inputs["input"] = tn.input_ports["x"]
            # Add module to graph and bind it input port 'x'.
            y = tn(x=model.inputs["input"])
            # Manual output bind.
            # model.outputs["output"] = y # BUG!

        # Serialize "model".
        serialized_model = model.serialize()

        # Delete model-related stuff.
        del tn
        del model

        # Deserialize the "model copy".
        model_copy = NeuralGraph.deserialize(serialized_model, name="model_copy")        

        # Build the "training graph" - using the model copy.
        with NeuralGraph(operation_mode=OperationMode.training, name="training") as training:
            # Add modules to graph.
            x, t = dl()
            # Incorporate modules from the existing "model" graph.
            p = model_copy(input=x)
            lss = loss(predictions=p, target=t)

        # Serialize the "training graph".
        serialized_training = training.serialize()
        
        # Delete everything.
        del dl
        del loss
        del model_copy
        del training

        # Create the second graph - deserialize withoput "module reusing".
        training2 = NeuralGraph.deserialize(serialized_training)
        serialized_training2 = training2.serialize()

        # import pdb;pdb.set_trace()
        # print("1: \n",serialized_training)
        # print("2: \n",serialized_training2)

        # Must be the same.
        assert serialized_training == serialized_training2
