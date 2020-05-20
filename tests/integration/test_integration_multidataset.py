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

import os
import shutil
from unittest import TestCase

import pytest
import torch

import nemo
from nemo.backends.pytorch.common import DataCombination
from nemo.core import ChannelType, NeuralType
from nemo.utils import logging


@pytest.mark.usefixtures("neural_factory")
class TestMultiDLIntegration(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    @pytest.mark.integration
    def test_pipeline(self):
        batch_size = 4
        dataset_size_0 = 100
        dataset_size_1 = 100
        shuffle = False
        dl_1 = nemo.backends.pytorch.tutorials.RealFunctionDataLayer(batch_size=batch_size, n=dataset_size_0)
        dl_2 = nemo.backends.pytorch.tutorials.RealFunctionDataLayer(batch_size=batch_size, n=dataset_size_1)

        data_layer = nemo.backends.pytorch.common.MultiDataLayer(
            data_layers=[dl_1, dl_2], batch_size=batch_size, shuffle=shuffle, combination_mode=DataCombination.ZIP
        )
        x_0, y_0, x_1, y_1 = data_layer()

        trainable_module = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        loss = nemo.backends.pytorch.tutorials.MSELoss()
        combined_loss = nemo.backends.pytorch.common.losses.LossAggregatorNM(num_inputs=2)
        pred_0 = trainable_module(x=x_0)
        pred_1 = trainable_module(x=x_1)
        l_0 = loss(predictions=pred_0, target=y_0)
        l_1 = loss(predictions=pred_1, target=y_1)
        total_loss = combined_loss(loss_1=l_0, loss_2=l_1)

        callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[total_loss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}'),
        )
        # Instantiate an optimizer to perform `train` action
        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            tensors_to_optimize=[total_loss], optimizer="sgd", optimization_params={"lr": 0.0003, "max_steps": 2},
        )
