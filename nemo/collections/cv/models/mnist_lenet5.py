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

from dataclasses import dataclass

from torch.utils.data import DataLoader

from nemo.collections.cv.datasets import MNISTDataset, MNISTDatasetConfig
from nemo.collections.cv.models import LeNet5, LeNet5Config
from nemo.core.config import DataLoaderConfig, NovogradConfig
from nemo.utils.decorators import experimental

__all__ = ['MNISTLeNet5', 'MNISTLeNet5Config']


@dataclass
class MNISTLeNet5Config(LeNet5Config):
    """
    Structured config for LeNet-5 model class - that also contains parameters of dataset and dataloader.

    (This example shows that we can inherit from other configs.)

    Args:
        opt: Lets use Novograd optimizer.
    """

    opt: NovogradConfig = NovogradConfig()
    dataset: MNISTDatasetConfig = MNISTDatasetConfig(width=32, height=32)
    dataloader: DataLoaderConfig = DataLoaderConfig(batch_size=64, shuffle=True)


@experimental
class MNISTLeNet5(LeNet5):
    def __init__(self, cfg: MNISTLeNet5Config = MNISTLeNet5Config()):
        super().__init__(cfg)

    def train_dataloader(self):
        """ Create dataset, wrap it with dataloader and return the latter """
        # Instantiate dataset.
        mnist_ds = MNISTDataset(self._cfg.dataset)
        # Configure data loader.
        train_dataloader = DataLoader(dataset=mnist_ds, **(self._cfg.dataloader))

        return train_dataloader
