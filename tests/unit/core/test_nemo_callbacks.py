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
from io import StringIO

import pytest
from tensorboard.backend.event_processing import event_file_inspector as efi
from torch.utils.tensorboard import SummaryWriter

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.backends.pytorch.tutorials import MSELoss, RealFunctionDataLayer, TaylorNet
from nemo.core.callbacks import *
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils import logging


@pytest.mark.usefixtures("neural_factory")
class TestNeMoCallbacks:
    @pytest.fixture()
    def clean_up(self):
        yield
        self.nf.reset_trainer()

    @pytest.mark.unit
    def test_SimpleLogger(self, clean_up):
        data_source = RealFunctionDataLayer(n=100, batch_size=1)
        trainable_module = TaylorNet(dim=4)
        loss = MSELoss()

        # Create the graph by connnecting the modules.
        x, y = data_source()
        y_pred = trainable_module(x=x)
        loss_tensor = loss(predictions=y_pred, target=y)

        # Mock up both std and stderr streams.
        with logging.patch_stdout_handler(StringIO()) as std_out:
            self.nf.train(
                tensors_to_optimize=[loss_tensor],
                callbacks=[SimpleLogger(step_freq=1)],
                optimization_params={"max_steps": 4, "lr": 0.01},
                optimizer="sgd",
            )

        output_lines = std_out.getvalue().splitlines()
        assert len(output_lines) == 4
        for line in output_lines:
            assert "loss" in line

    @pytest.mark.unit
    def test_rename_and_log(self, clean_up):
        data_source = RealFunctionDataLayer(n=100, batch_size=1)
        trainable_module = TaylorNet(dim=4)
        loss = MSELoss()

        # Create the graph by connnecting the modules.
        x, y = data_source()
        y_pred = trainable_module(x=x)
        loss_tensor = loss(predictions=y_pred, target=y)

        class DummyNM(NonTrainableNM):
            def __init__(self):
                super().__init__()

            @property
            def input_ports(self):
                """Returns definitions of module input ports.

                Returns:
                  A (dict) of module's input ports names to NeuralTypes mapping
                """
                return {"x": NeuralType(('B', 'D'), ChannelType())}

            @property
            def output_ports(self):
                """Returns definitions of module output ports.

                Returns:
                  A (dict) of module's output ports names to NeuralTypes mapping
                """
                return {"y_pred": NeuralType(('B', 'D'), ChannelType())}

            def forward(self, x):
                return x + 1

        test = DummyNM()
        extra_tensor = test(x=y_pred)

        y_pred.rename("y_pred")
        assert y_pred.name == "y_pred"

        # Mock up both std and stderr streams.
        with logging.patch_stdout_handler(StringIO()) as std_out:
            self.nf.train(
                tensors_to_optimize=[loss_tensor],
                callbacks=[SimpleLogger(step_freq=1, tensors_to_log=['y_pred', extra_tensor])],
                optimization_params={"max_steps": 4, "lr": 0.01},
                optimizer="sgd",
            )

        output_lines = std_out.getvalue().splitlines()
        assert len(output_lines) == 8
        for i, line in enumerate(output_lines):
            if i % 2 == 0:
                assert y_pred.name in line
            else:
                assert extra_tensor.name in line

    @pytest.mark.unit
    def test_TensorboardLogger(self, clean_up, tmpdir):
        data_source = RealFunctionDataLayer(n=100, batch_size=1)
        trainable_module = TaylorNet(dim=4)
        loss = MSELoss()

        # Create the graph by connnecting the modules.
        x, y = data_source()
        y_pred = trainable_module(x=x)
        loss_tensor = loss(predictions=y_pred, target=y)

        logging_dir = tmpdir.mkdir("temp")

        writer = SummaryWriter(logging_dir)

        tb_logger = TensorboardLogger(writer, step_freq=1)
        callbacks = [tb_logger]

        self.nf.train(
            tensors_to_optimize=[loss_tensor],
            callbacks=callbacks,
            optimization_params={"max_steps": 4, "lr": 0.01},
            optimizer="sgd",
        )

        # efi.inspect("temp", tag="loss")
        inspection_units = efi.get_inspection_units(str(logging_dir), "", "loss")

        # Make sure there is only 1 tensorboard file
        assert len(inspection_units) == 1

        # Assert that there the loss scalars has been logged 4 times
        assert len(inspection_units[0].field_to_obs['scalars']) == 4

    @pytest.mark.unit
    def test_epoch_decorators(self, clean_up):
        data_source = RealFunctionDataLayer(n=24, batch_size=12)
        trainable_module = TaylorNet(dim=4)
        loss = MSELoss()

        # Create the graph by connnecting the modules.
        x, y = data_source()
        y_pred = trainable_module(x=x)
        loss_tensor = loss(predictions=y_pred, target=y)

        epoch_start_counter = [0]
        epoch_end_counter = [0]

        @on_epoch_start
        def count_epoch_starts(state, counter=epoch_start_counter):
            counter[0] += 1

        @on_epoch_end
        def count_epoch_ends(state, counter=epoch_end_counter):
            counter[0] -= 1

        callbacks = [count_epoch_starts, count_epoch_ends]

        self.nf.train(
            tensors_to_optimize=[loss_tensor],
            callbacks=callbacks,
            optimization_params={"max_steps": 4, "lr": 0.01},
            optimizer="sgd",
        )

        assert epoch_start_counter[0] == 2
        assert epoch_end_counter[0] == -2

    @pytest.mark.unit
    def test_step_batch_decorators(self, clean_up):
        """Showcase the difference between step and batch"""
        data_source = RealFunctionDataLayer(n=24, batch_size=12)
        trainable_module = TaylorNet(dim=4)
        loss = MSELoss()

        # Create the graph by connnecting the modules.
        x, y = data_source()
        y_pred = trainable_module(x=x)
        loss_tensor = loss(predictions=y_pred, target=y)

        epoch_step_counter = [0]
        epoch_batch_counter = [0]

        @on_step_end
        def count_steps(state, counter=epoch_step_counter):
            counter[0] += 1

        @on_batch_end
        def count_batches(state, counter=epoch_batch_counter):
            counter[0] += 1

        callbacks = [count_steps, count_batches]

        self.nf.train(
            tensors_to_optimize=[loss_tensor],
            callbacks=callbacks,
            optimization_params={"max_steps": 4, "lr": 0.01},
            optimizer="sgd",
        )

        # when grad accumlation steps (aka iter_per_step or batches_per_step) = 1, num_steps == num_batches
        assert epoch_step_counter[0] == 4
        assert epoch_batch_counter[0] == 4

        epoch_step_counter[0] = 0
        epoch_batch_counter[0] = 0

        self.nf.train(
            tensors_to_optimize=[loss_tensor],
            callbacks=callbacks,
            optimization_params={"max_steps": 4, "lr": 0.01},
            optimizer="sgd",
            reset=True,
            batches_per_step=2,
        )

        # when grad accumlation steps != 1, num_steps != num_batches
        assert epoch_step_counter[0] == 4
        assert epoch_batch_counter[0] == 8
