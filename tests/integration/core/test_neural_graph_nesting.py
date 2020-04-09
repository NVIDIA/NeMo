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
from nemo.utils import logging


@pytest.mark.usefixtures("neural_factory")
class TestNeuralGraphNesting:
    @pytest.mark.integration
    def test_nesting_operation_modes_ok(self):
        """ 
            Tests whether one can nest one graph in mode `both` (representing the our `model`) into
            `training` and validation (`inference`) graphs.
        """
        # Instantiate the necessary neural modules.
        dl_training = RealFunctionDataLayer(n=100, batch_size=4)
        dl_validation = RealFunctionDataLayer(n=100, batch_size=4)
        fx = TaylorNet(dim=4)
        loss = MSELoss()

        with NeuralGraph(operation_mode=OperationMode.both) as model:
            # Bind the input.
            _ = fx(x=model)
            # All outputs will be bound by default.

        # Nest model into training graph.
        with NeuralGraph(operation_mode=OperationMode.training) as training_graph:
            # Take outputs from the training DL.
            x, t = dl_training()
            # Pass them to the model
            p = model(x=x)
            # Pass both of them to loss.
            lss = loss(predictions=p, target=t)

        # Nest model into validation graph.
        with NeuralGraph(operation_mode=OperationMode.inference) as validation_graph:
            # Take outputs from the training DL.
            x_valid, t_valid = dl_training()
            # Pass them to the model
            p_valid = model(x=x_valid)
            loss_e = loss(predictions=p_valid, target=t_valid)

        # Callbacks to print info to console and Tensorboard.
        train_callback = SimpleLossLoggerCallback(
            tensors=[lss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}')
        )

        def batch_loss_per_batch_callback(tensors, global_vars):
            if "batch_loss" not in global_vars.keys():
                global_vars["batch_loss"] = []
            for key, value in tensors.items():
                if key.startswith("loss"):
                    global_vars["batch_loss"].append(torch.mean(torch.stack(value)))

        def batch_loss_epoch_finished_callback(global_vars):
            epoch_loss = torch.max(torch.tensor(global_vars["batch_loss"]))
            logging.info("Evaluation Loss: {0}".format(epoch_loss))
            return dict({"Evaluation Loss": epoch_loss})

        eval_callback = EvaluatorCallback(
            eval_tensors=[loss_e],
            user_iter_callback=batch_loss_per_batch_callback,
            user_epochs_done_callback=batch_loss_epoch_finished_callback,
            eval_step=1,
        )

        # Instantiate an optimizer to perform the `train` action.
        optimizer = PtActions()
        # Invoke "train" action - perform single forward-backard step.
        optimizer.train(
            [lss],
            callbacks=[train_callback, eval_callback],
            optimization_params={"max_steps": 2, "lr": 0.0003},
            optimizer="sgd",
        )
