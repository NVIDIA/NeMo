# ! /usr/bin/python
# -*- coding: utf-8 -*-

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

import nemo
from tests.common_setup import NeMoUnitTest

logging = nemo.logging


class TestPytorchTrainers(NeMoUnitTest):
    def test_simple_train(self):
        logging.info("Simplest train test")
        data_source = nemo.backends.pytorch.tutorials.RealFunctionDataLayer(n=10000, batch_size=128)
        trainable_module = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        loss = nemo.backends.pytorch.tutorials.MSELoss()
        x, y = data_source()
        y_pred = trainable_module(x=x)
        loss_tensor = loss(predictions=y_pred, target=y)

        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            tensors_to_optimize=[loss_tensor], optimizer="sgd", optimization_params={"lr": 0.0003, "num_epochs": 1},
        )

    def test_simple_train_named_output(self):
        logging.info('Simplest train test with using named output.')
        data_source = nemo.backends.pytorch.tutorials.RealFunctionDataLayer(n=10000, batch_size=128,)
        trainable_module = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        loss = nemo.backends.pytorch.tutorials.MSELoss()

        data = data_source()
        self.assertEqual(
            first=type(data).__name__,
            second='RealFunctionDataLayerOutput',
            msg='Check output class naming coherence.',
        )
        y_pred = trainable_module(x=data.x)
        loss_tensor = loss(predictions=y_pred, target=data.y)

        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            tensors_to_optimize=[loss_tensor], optimizer="sgd", optimization_params={"lr": 0.0003, "num_epochs": 1},
        )

    def test_simple_chained_train(self):
        logging.info("Chained train test")
        data_source = nemo.backends.pytorch.tutorials.RealFunctionDataLayer(n=10000, batch_size=32)
        trainable_module1 = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        trainable_module2 = nemo.backends.pytorch.tutorials.TaylorNet(dim=2)
        trainable_module3 = nemo.backends.pytorch.tutorials.TaylorNet(dim=2)
        loss = nemo.backends.pytorch.tutorials.MSELoss()
        x, y = data_source()
        y_pred1 = trainable_module1(x=x)
        y_pred2 = trainable_module2(x=y_pred1)
        y_pred3 = trainable_module3(x=y_pred2)
        loss_tensor = loss(predictions=y_pred3, target=y)

        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            tensors_to_optimize=[loss_tensor], optimizer="sgd", optimization_params={"lr": 0.0003, "num_epochs": 1},
        )
