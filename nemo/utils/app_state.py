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

from nemo.utils.metaclasses import Singleton


class AppState(metaclass=Singleton):
    def __init__(self):

        # TODO: should we store global config in hydra_runner?
        self._app_cfg = None

        # World info
        self._device_id = None
        self._local_rank = None
        self._global_rank = None
        self._model_parallel_rank = None
        self._data_parallel_rank = None

        self._world_size = None
        self._model_parallel_size = None
        self._model_parallel_group = None
        self._data_parallel_size = None
        self._data_parallel_group = None

        self._random_seed = None

    @property
    def device_id(self):
        """ Property returns the device_id
            Returns:
                device_id
        """
        return self._device_id

    @device_id.setter
    def device_id(self, id):
        """ Property sets the device_id.
            Args:
                size (int): The device id. 
        """
        self._device_id = id

    @property
    def world_size(self):
        """ Property returns the total number of GPUs.
            Returns:
                Total number of GPUs.
        """
        return self._world_size

    @world_size.setter
    def world_size(self, size):
        """ Property sets the total number of GPUs.
            Args:
                size (int):  Total number of GPUs.
        """
        self._world_size = size

    @property
    def model_parallel_size(self):
        """ Property returns the number of GPUs in each model parallel group.
            Returns:
                Number of GPUs in each model parallel group.
        """
        return self._model_parallel_size

    @model_parallel_size.setter
    def model_parallel_size(self, size):
        """ Property sets the number of GPUs in each model parallel group.
            Args:
                size (int):  Number of GPUs in each model parallel group.
        """
        self._model_parallel_size = size

    @property
    def data_parallel_size(self):
        """ Property returns the number of GPUs in each data parallel group.
            Returns:
                Number of GPUs in each data parallel group.
        """
        return self._data_parallel_size

    @data_parallel_size.setter
    def data_parallel_size(self, size):
        """ Property sets the number of GPUs in each data parallel group.
            Args:
                size (int):  Number of GPUs in each data parallel group.
        """
        self._data_parallel_size = size

    @property
    def local_rank(self):
        """ Property returns the local rank.
            Returns:
                Local rank.
        """
        return self._local_rank

    @local_rank.setter
    def local_rank(self, rank):
        """ Property sets the local rank.
            Args:
                rank (int):  Local rank.
        """
        self._local_rank = rank

    @property
    def global_rank(self):
        """ Property returns the global rank.
            Returns:
                Global rank.
        """
        return self._global_rank

    @global_rank.setter
    def global_rank(self, rank):
        """ Property sets the global rank.
            Args:
                rank (int):  Global rank.
        """
        self._global_rank = rank

    @property
    def model_parallel_rank(self):
        """ Property returns the model parallel rank.
            Returns:
                Model parallel rank.
        """
        return self._model_parallel_rank

    @model_parallel_rank.setter
    def model_parallel_rank(self, rank):
        """ Property sets the model parallel rank.
            Args:
                rank (int):  Model parallel rank.
        """
        self._model_parallel_rank = rank

    @property
    def model_parallel_group(self):
        """ Property returns the model parallel group.
            Returns:
                Model parallel group.
        """
        return self._model_parallel_group

    @model_parallel_group.setter
    def model_parallel_group(self, group):
        """ Property sets the model parallel group.
            Args:
                group:  Model parallel group.
        """
        self._model_parallel_group = group

    @property
    def data_parallel_rank(self):
        """ Property returns the data parallel rank.
            Returns:
                Data parallel rank.
        """
        return self._data_parallel_rank

    @data_parallel_rank.setter
    def data_parallel_rank(self, rank):
        """ Property sets the data parallel rank.
            Args:
                rank (int):  Data parallel rank.
        """
        self._data_parallel_rank = rank

    @property
    def data_parallel_group(self):
        """ Property returns the data parallel group.
            Returns:
                Data parallel group.
        """
        return self._data_parallel_group

    @data_parallel_group.setter
    def data_parallel_group(self, group):
        """ Property sets the data parallel group.
            Args:
                group:  Data parallel group.
        """
        self._data_parallel_group = group

    @property
    def random_seed(self):
        """ Property returns the random seed.
            Returns:
                Random seed.
        """
        return self._random_seed

    @random_seed.setter
    def random_seed(self, seed):
        """ Property sets the random seed.
            Args:
                seed (int):  Random seed.
        """
        self._random_seed = seed
