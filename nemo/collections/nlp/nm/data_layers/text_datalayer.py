# =============================================================================
# Copyright 2019 AI Applications Design Team at NVIDIA. All Rights Reserved.
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
        dataset_type: type of dataset used for this datalayer
        dataset_params (dict): all the params for the dataset
    """

    def __init__(self, dataset_type, dataset_params, batch_size, shuffle):
        super().__init__()
        self._dataset = dataset_type(**dataset_params)
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None
