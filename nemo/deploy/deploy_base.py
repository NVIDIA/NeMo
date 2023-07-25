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
import torch
from nemo.core.classes.modelPT import ModelPT
from nemo.deploy.triton_deployable import ITritonDeployable
import importlib

from pytorch_lightning import Trainer
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

from abc import ABC, abstractmethod


class DeployBase(ABC):

    def __init__(self,
                 checkpoint_path: str,
                 triton_model_name: str,
                 max_batch_size: int=128,
    ):
        self.checkpoint_path = checkpoint_path
        self.triton_model_name = triton_model_name
        self.max_batch_size = max_batch_size
        self.model = None
        self.triton = None

    @abstractmethod
    def deploy(self):
        pass

    @abstractmethod
    def serve(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def _init_nemo_model(self):
        model_config = ModelPT.restore_from(self.checkpoint_path, return_config=True)
        module_path, class_name = DeployBase.get_module_and_class(model_config.target)
        cls = getattr(importlib.import_module(module_path), class_name)
        self.model = cls.restore_from(restore_path=self.checkpoint_path, trainer=Trainer())

        if not issubclass(type(self.model), ITritonDeployable):
            raise Exception("This model is not deployable to Triton."
                            "nemo.deploy.ITritonDeployable class should be inherited")

    @staticmethod
    def get_module_and_class(target: str):
        l = target.rindex(".")
        return target[0:l], target[l + 1:len(target)]