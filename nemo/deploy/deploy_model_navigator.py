# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
from pytriton.decorators import batch
import torch
from nemo.core.classes.modelPT import ModelPT
from .deploy_base import DeployBase
import importlib

from pytorch_lightning import Trainer
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton


class DeployModelNavigator(DeployBase):

    def __init__(self,
                 checkpoint_path: str,
                 triton_model_name: str,
                 max_batch_size: int=128,
    ):
        super().__init__(checkpoint_path=checkpoint_path,
                         triton_model_name=triton_model_name,
                         max_batch_size=max_batch_size,
                    )

    def deploy(self):
        raise NotImplementedError

    def serve(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError
