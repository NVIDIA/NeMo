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

import importlib
from abc import ABC, abstractmethod

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

from nemo.core.classes.modelPT import ModelPT
from nemo.deploy.triton_deployable import ITritonDeployable


class DeployBase(ABC):
    def __init__(
        self,
        triton_model_name: str,
        triton_model_version: int = 1,
        checkpoint_path: str = None,
        model=None,
        max_batch_size: int = 128,
    ):
        self.checkpoint_path = checkpoint_path
        self.triton_model_name = triton_model_name
        self.triton_model_version = triton_model_version
        self.max_batch_size = max_batch_size
        self.model = model
        self.triton = None

        if checkpoint_path is None and model is None:
            raise Exception("Either checkpoint_path or model should be provided.")

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
        if self.checkpoint_path is not None:
            model_config = ModelPT.restore_from(self.checkpoint_path, return_config=True)
            module_path, class_name = DeployBase.get_module_and_class(model_config.target)
            cls = getattr(importlib.import_module(module_path), class_name)
            self.model = cls.restore_from(restore_path=self.checkpoint_path, trainer=Trainer())
            self.model.freeze()

            # has to turn off activations_checkpoint_method for inference
            try:
                self.model.model.language_model.encoder.activations_checkpoint_method = None
            except AttributeError:
                pass

        if self.model is None:
            raise Exception("There is no model to deploy.")

        self._is_model_deployable()

    def _is_model_deployable(self):
        if not issubclass(type(self.model), ITritonDeployable):
            raise Exception(
                "This model is not deployable to Triton." "nemo.deploy.ITritonDeployable class should be inherited"
            )
        else:
            return True

    @staticmethod
    def get_module_and_class(target: str):
        l = target.rindex(".")
        return target[0:l], target[l + 1 : len(target)]
