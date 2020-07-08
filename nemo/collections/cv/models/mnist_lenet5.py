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

from typing import Dict

from torch.utils.data import DataLoader

from nemo.collections.cv.models import LeNet5
from nemo.collections.cv.datasets import MNISTDataset

from nemo.utils.decorators import experimental


__all__ = ['MNISTLeNet5']


@experimental
class MNISTLeNet5(LeNet5):
    def __init__(
        self,
        params: Dict,
    ):
        super().__init__(params)
    
    
    def train_dataloader(self):
        """ Create dataset, wrap it with dataloader and return the latter """
        # Instantiate Dataset.
        mnist_ds = MNISTDataset(height=32, width=32, train=True)
        # Configure data loader.
        train_dataloader = DataLoader(dataset=mnist_ds, batch_size=128, shuffle=True)
        return train_dataloader
