# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, Union

import torch
from omegaconf import DictConfig

from nemo.core.classes import ModelPT


class MockModel(ModelPT):
    def __init__(self, cfg, trainer=None):
        super(MockModel, self).__init__(cfg=cfg, trainer=trainer)
        self.w = torch.nn.Linear(10, 1)
        # mock temp file
        if 'temp_file' in self.cfg and self.cfg.temp_file is not None:
            self.temp_file = self.register_artifact('temp_file', self.cfg.temp_file)
            with open(self.temp_file, 'r') as f:
                self.temp_data = f.readlines()
        else:
            self.temp_file = None
            self.temp_data = None

    def forward(self, x):
        y = self.w(x)
        return y, self.cfg.temp_file

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        self._train_dl = None

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self._validation_dl = None

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self._test_dl = None

    def list_available_models(cls):
        return []
