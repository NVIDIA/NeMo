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

from typing import Any, Dict, Optional, Union

import hydra
from omegaconf import OmegaConf, DictConfig

import torchvision.transforms as transforms

from nemo.core.classes import ModelPT
from nemo.core.config import DataLoaderConfig


class Model(ModelPT):
    """
    The parent class of all CV models.
    Implements some basic functionality reused in all models plus provides empty implementations of methods that are
    not required for proper functioning of the model.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)

    @classmethod
    def instantiate_dataloader(
        cls, dataloader_cfg: DataLoaderConfig, dataset_cfg: DictConfig, transform_cfg: DictConfig = None
    ):
        """
        Creates the dataset and wraps it with a dataloader.
        Optionally, creates a set of transforms and passes them to dataset.

        Args:
            dataloader_cfg: Configuration of PyTorch DataLoader
            dataset_cfg: Configuration of the dataset
            transform_cfg: Configuration of transforms (Optional, Default: None)
        """
        # Check transforms configuration.
        if transform_cfg is not None:
            transform = transforms.Compose([hydra.utils.instantiate(trans) for trans in transform_cfg])
        else:
            transform = None

        # Instantiate dataset.
        ds = hydra.utils.instantiate(dataset_cfg, transform=transform)
        # Instantiate dataloader.
        dl = hydra.utils.instantiate(dataloader_cfg, dataset=ds)

        # Return the dataloader.
        return dl

    def setup_optimization(self, optim_cfg: DictConfig):
        """
        Sets up the optimization (optimizers, schedulers etc.).
        """
        # Instantiate optimizer.
        self._optim = hydra.utils.instantiate(optim_cfg, params=self.parameters())

        # Test of idea - dynamically add of method returning optimizer.
        # Goal: keeping the original "PTL safety mechanisms" when one does not create optimizers =>
        # trainer will act in a standard way.
        def configure_optimizers_dynamic(self):
            return self._optim

        # Add it/overwrite the class method.
        setattr(self.__class__, 'configure_optimizers', configure_optimizers_dynamic)

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        """ Not implemented. """
        return {}

    def setup_training_data(self, val_ds_config: Optional[Dict] = None):
        """ Not implemented. """
        pass

    def setup_validation_data(self, val_ds_config: Optional[Dict] = None):
        """ Not implemented. """
        pass
