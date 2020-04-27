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

from enum import Enum
from typing import List

import numpy as np
import torch

from nemo import logging
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import *

__all__ = ['MultiDataLayer', 'DataCombination']


class DataCombination(Enum):
    CROSSPRODUCT = 1
    ZIP = 2


class MultiDataLayer(DataLayerNM):
    def __init__(
        self,
        data_layers: List[DataLayerNM],
        batch_size: int,
        shuffle: bool = False,
        combination_mode: DataCombination = DataCombination.CROSSPRODUCT,
        port_names: List[str] = None,
    ):
        """
        data_layers: (list) of DataLayerNM objects
        batch_size: (int) batchsize when the underlying dataset is loaded
        combination_mode: (DataCombination) defines how to combine the datasets.
        shuffle: (bool) whether underlying multi dataset should be shuffled in each epoch
        port_names: List(str) user can override all port names if specified 
        """
        super().__init__()
        self._data_layers = data_layers
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._combination_mode = combination_mode
        self._port_names = port_names
        self._dataset = MultiDataset(
            datasets=[dl.dataset for dl in self._data_layers], combination_mode=combination_mode
        )

        self._ports = dict()
        if self._port_names:
            i = 0
            for dl in self._data_layers:
                for _, port_type in dl.output_ports.items():
                    self._ports[self._port_names[i]] = port_type
                    i += 1
        else:
            for dl_idx, dl in enumerate(self._data_layers):
                for port_name, port_type in dl.output_ports.items():
                    if port_name in self._ports:
                        logging.warning(f"name collision {port_name}, will rename")
                        self._ports[f"{port_name}_{dl_idx}"] = port_type
                    else:
                        self._ports[port_name] = port_type

    @property
    def output_ports(self):
        """Return: dict
        Returns union of all individual data_layer output ports
        In case of name collision, resolve by renaming 
        """
        return self._ports

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None


class MultiDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: List[torch.utils.data.Dataset],
        combination_mode: DataCombination = DataCombination.CROSSPRODUCT,
    ):
        """
        Datasets: list of torch.utils.data.Dataset objects.
        combination_mode: DataCombination, defines how to combine the datasets, Options are [DataCombination.CROSSPRODUCT, DataCombination.ZIP]. 
        """
        self.datasets = datasets
        self.combination_mode = combination_mode
        if self.combination_mode == DataCombination.CROSSPRODUCT:
            self.len = np.prod([len(d) for d in self.datasets])
        elif self.combination_mode == DataCombination.ZIP:
            ds_lens = [len(d) for d in self.datasets]
            self.len = np.min(ds_lens)
            if len(set(ds_lens)) != 1:
                raise ValueError("datasets do not have equal lengths.")
        else:
            raise ValueError("combination_mode unknown")

    def __getitem__(self, i):
        """
        Returns list [x1, x2, ...xn] where x1 \in D1, x2 \in D2, ..., xn \in Dn
        """

        return [x for d in self.datasets for x in d[i % len(d)]]

    def __len__(self):
        """
        Returns length of this dataset (int).
        In case of  DataCombination.CROSSPRODUCT this would be prod(len(d) for d in self.datasets). 
        In case of  DataCombination.ZIP this would be min(len(d) for d in self.datasets) given that all datasets have same length. 
        """
        return self.len
