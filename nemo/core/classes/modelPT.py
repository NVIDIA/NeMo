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

from abc import abstractmethod
from typing import Dict, Optional

from pytorch_lightning import LightningModule
from torch.optim.optimizer import Optimizer

from nemo.core.classes.common import Model
from nemo.core.optim.optimizers import get_optimizer, parse_optimizer_args
from nemo.utils import logging

__all__ = ['ModelPT']


class ModelPT(LightningModule, Model):
    """
    Interface for Pytorch-lightning based NeMo models
    """

    @abstractmethod
    def setup_training_data(self, train_data_layer_params: Optional[Dict]):
        """
        Setups data loader to be used in training
        Args:
            train_data_layer_params: training data layer parameters.
        Returns:

        """
        pass

    @abstractmethod
    def setup_validation_data(self, val_data_layer_params: Optional[Dict]):
        """
        (Optionally) Setups data loader to be used in validation
        Args:
            val_data_layer_params: validation data layer parameters.
        Returns:

        """
        pass

    def setup_optimization(self, optim_params: Optional[Dict] = None) -> Optimizer:
        """
        Prepares an optimizer from a string name and its optional config parameters.

        Args:
            optim_params: a dictionary containing the following keys.
                - "lr": mandatory key for learning rate. Will raise ValueError
                if not provided.

                - "optimizer": string name pointing to one of the available
                optimizers in the registry. If not provided, defaults to "adam".

                - "opt_args": Optional list of strings, in the format "arg_name=arg_value".
                The list of "arg_value" will be parsed and a dictionary of optimizer
                kwargs will be built and supplied to instantiate the optimizer.

        Returns:
            An instance of a torch.optim.Optimizer
        """
        optim_params = optim_params or {}  # In case null was passed as optim_params

        # Check if caller provided optimizer name, default to Adam otherwise
        optimizer_name = optim_params.get('optimizer', 'adam')

        # Check if caller has optimizer kwargs, default to empty dictionary
        optimizer_args = optim_params.get('opt_args', [])
        optimizer_args = parse_optimizer_args(optimizer_args)

        # We are guarenteed to have lr since it is required by the argparser
        # But maybe user forgot to pass it to this function
        lr = optim_params.get('lr', None)

        if 'lr' is None:
            raise ValueError('`lr` must be passed to `optim_params` when setting up the optimization !')

        # Actually instantiate the optimizer
        optimizer = get_optimizer(optimizer_name)
        optimizer = optimizer(self.parameters(), lr=lr, **optimizer_args)

        # TODO: Remove after demonstration
        logging.info("Optimizer config = %s", str(optimizer))

        return optimizer
