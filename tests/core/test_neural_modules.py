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
from nemo.core.neural_types import ChannelType, NeuralType
from tests.common_setup import NeMoUnitTest


class NeuralModulesTests(NeMoUnitTest):
    def test_call_TaylorNet(self):
        x_tg = nemo.core.neural_modules.NmTensor(
            producer=None, producer_args=None, name=None, ntype=NeuralType(('B', 'D'), ChannelType())
        )

        tn = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        # note that real port's name: x was used
        y_pred = tn(x=x_tg)
        self.assertEqual(y_pred.producer, tn)
        self.assertEqual(y_pred.producer_args.get("x"), x_tg)

    def test_simplest_example_chain(self):
        data_source = nemo.backends.pytorch.tutorials.RealFunctionDataLayer(n=10000, batch_size=1)
        trainable_module = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        loss = nemo.backends.pytorch.tutorials.MSELoss()
        x, y = data_source()
        y_pred = trainable_module(x=x)
        loss_tensor = loss(predictions=y_pred, target=y)

        # check producers' bookkeeping
        self.assertEqual(loss_tensor.producer, loss)
        self.assertEqual(loss_tensor.producer_args, {"predictions": y_pred, "target": y})
        self.assertEqual(y_pred.producer, trainable_module)
        self.assertEqual(y_pred.producer_args, {"x": x})
        self.assertEqual(y.producer, data_source)
        self.assertEqual(y.producer_args, {})
        self.assertEqual(x.producer, data_source)
        self.assertEqual(x.producer_args, {})
