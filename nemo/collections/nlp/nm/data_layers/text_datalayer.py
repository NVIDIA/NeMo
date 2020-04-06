# =============================================================================
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

from nemo.backends.pytorch import DataLayerNM
from nemo.collections.nlp.data.datasets import *

__all__ = ['TextDataLayer']


class TextDataLayer(DataLayerNM):
    """
    Generic Text Data Layer NM which wraps PyTorch's dataset

    Args:
        dataset_type (Dataset): type of dataset used for this datalayer
        dataset_params (dict): all the params for the dataset
        batch_size (int): sequence batch size
        shuffle (bool): whether to shuffle data
    """

    def __init__(self, dataset_type, dataset_params, batch_size, shuffle=False, num_workers=-1, pin_memory=False):
        super().__init__()
        self._dataset = dataset_type(**dataset_params)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._pin_memory = pin_memory
        if num_workers >= 0:
            self._num_workers = num_workers

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None
