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


__all__ = ['ModelPT']

from abc import abstractmethod
from typing import Dict, Optional

from pytorch_lightning import LightningModule

from nemo.core.classes.common import Model
from nemo.core.classes.common import NeMoModel


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

    @abstractmethod
    def setup_test_data(self, test_data_layer_params: Optional[Dict]):
        """
        (Optionally) Setups data loader to be used in testing
        Args:
            test_data_layer_params: test data layer parameters.
        Returns:

        """
        pass

    @abstractmethod
    def setup_optimization(self, optim_params: Optional[Dict]):
        """
        Setups optimization parameters
        Args:
            optim_params: dictionary with optimization parameters.
        Returns:

        """
        pass
