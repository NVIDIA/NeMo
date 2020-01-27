# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2019 NVIDIA. All Rights Reserved.
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

import nemo
from .common_setup import NeMoUnitTest


class TestNeuralFactory(NeMoUnitTest):
    def test_creation(self):
        neural_factory = nemo.core.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch, local_rank=None, create_tb_writer=False,
        )
        instance = neural_factory.get_module(name="TaylorNet", collection="toys", params={"dim": 4})
        self.assertTrue(isinstance(instance, nemo.backends.pytorch.tutorials.TaylorNet))

    def test_simple_example(self):
        neural_factory = nemo.core.neural_factory.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch, local_rank=None, create_tb_writer=False,
        )
        dl = neural_factory.get_module(
            name="RealFunctionDataLayer", collection="toys", params={"n": 10000, "batch_size": 128},
        )
        fx = neural_factory.get_module(name="TaylorNet", collection="toys", params={"dim": 4})
        loss = neural_factory.get_module(name="MSELoss", collection="toys", params={})

        x, y = dl()
        y_pred = fx(x=x)
        loss_tensor = loss(predictions=y_pred, target=y)

        optimizer = neural_factory.get_trainer()
        optimizer.train(
            [loss_tensor], optimizer="sgd", optimization_params={"lr": 1e-3, "num_epochs": 1},
        )
