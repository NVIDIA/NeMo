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
from nemo.core.neural_types import NeuralTypeComparisonResult


@pytest.mark.usefixtures("neural_factory")
class TestNmTensor:
    @pytest.mark.unit
    def test_nm_tensors_producer_args(self):
        """
            Tests whether nmTensors are correct - producers and their args.
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
        assert loss_tensor.producer_name == loss.name
        assert loss_tensor.producer_args == {"predictions": y_pred, "target": y}
        assert y_pred.producer_name == trainable_module.name
        assert y_pred.producer_args == {"x": x}
        assert y.producer_name == data_source.name
        assert y.producer_args == {}
        assert x.producer_name == data_source.name
        assert x.producer_args == {}

    @pytest.mark.unit
    def test_simple_train_named_output(self):
        """ Test named output """
        data_source = RealFunctionDataLayer(n=10, batch_size=1)
        # Get data
        data = data_source()

        # Check output class naming coherence.
        assert type(data).__name__ == 'RealFunctionDataLayerOutput'

        # Check types.
        assert data.x.compare(data_source.output_ports["x"]) == NeuralTypeComparisonResult.SAME
        assert data.y.compare(data_source.output_ports["y"]) == NeuralTypeComparisonResult.SAME

    @pytest.mark.unit
    def test_nm_tensors_producer_consumers(self):
        """
            Tests whether nmTensors are correct - checking producers and consumers.
        """
        # Create modules.
        data_source = RealFunctionDataLayer(n=10, batch_size=1, name="source")
        trainable_module = TaylorNet(dim=4, name="tm")
        loss = MSELoss(name="loss")
        loss2 = MSELoss(name="loss2")

        # Create the graph by connnecting the modules.
        x, y = data_source()
        y_pred = trainable_module(x=x)
        lss = loss(predictions=y_pred, target=y)
        lss2 = loss2(predictions=y_pred, target=y)

        # Check tensor x producer and consumers.
        p = x.producer_step_module_port
        cs = x.consumers
        assert p.module_name == "source"
        assert p.port_name == "x"
        assert len(cs) == 1
        assert cs[0].module_name == "tm"
        assert cs[0].port_name == "x"

        # Check tensor y producer and consumers.
        p = y.producer_step_module_port
        cs = y.consumers
        assert p.module_name == "source"
        assert p.port_name == "y"
        assert len(cs) == 2
        assert cs[0].module_name == "loss"
        assert cs[0].port_name == "target"
        assert cs[1].module_name == "loss2"
        assert cs[1].port_name == "target"

        # Check tensor y_pred producer and consumers.
        p = y_pred.producer_step_module_port
        cs = y_pred.consumers
        assert p.module_name == "tm"
        assert p.port_name == "y_pred"
        assert len(cs) == 2
        assert cs[0].module_name == "loss"
        assert cs[0].port_name == "predictions"
        assert cs[1].module_name == "loss2"
        assert cs[1].port_name == "predictions"

    @pytest.mark.unit
    def test_nm_tensors_types(self):
        """
            Tests whether nmTensors are correct - checking type property.
        """
        # Create modules.
        data_source = RealFunctionDataLayer(n=10, batch_size=1)
        trainable_module = TaylorNet(dim=4)
        loss = MSELoss()

        # Create the graph by connnecting the modules.
        x, y = data_source()
        y_pred = trainable_module(x=x)
        lss = loss(predictions=y_pred, target=y)

        # Check types.
        assert x.ntype.compare(data_source.output_ports["x"]) == NeuralTypeComparisonResult.SAME
        assert y.ntype.compare(data_source.output_ports["y"]) == NeuralTypeComparisonResult.SAME
        assert y_pred.ntype.compare(trainable_module.output_ports["y_pred"]) == NeuralTypeComparisonResult.SAME
        assert lss.ntype.compare(loss.output_ports["loss"]) == NeuralTypeComparisonResult.SAME
