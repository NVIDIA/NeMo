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


class DeployPyTriton(DeployBase):

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
        self._init_nemo_model()
        try:
            self.triton = Triton()
            self.triton.bind(
                model_name=self.triton_model_name,
                infer_func=self.model.triton_infer_fn,
                inputs=self.model.get_triton_input_type,
                outputs=self.model.get_triton_output_type,
                config=ModelConfig(max_batch_size=self.max_batch_size)
            )
        except Exception as e:
            self.triton = None
            print(e)

    def serve(self):
        if self.triton is None:
            raise Exception("deploy should be called first.")

        try:
            self.triton.serve()
        except Exception as e:
            self.triton = None
            print(e)

    def run(self):
        if self.triton is None:
            raise Exception("deploy should be called first.")

        self.triton.run()

    def stop(self):
        if self.triton is None:
            raise Exception("deploy should be called first.")

        self.triton.stop()

    def _init_nemo_model(self):
        super()._init_nemo_model()

        if not hasattr(self.model, "triton_infer_fn"):
            raise Exception("triton_infer_fn function has not been found in the model class. "
                            "In order to use nemo deployment, model classes need to implement "
                            "triton_infer_fn function that runs inference.")

        if not hasattr(self.model, "get_triton_input_type"):
            raise Exception("get_triton_input_type function has not been found in the model class. "
                            "In order to use nemo deployment, model classes need to implement "
                            "get_triton_input_type function that returns the triton input types.")

        if not hasattr(self.model, "get_triton_output_type"):
            raise Exception("get_triton_output_type function has not been found in the model class. "
                            "In order to use nemo deployment, model classes need to implement "
                            "get_triton_output_type function that returns the triton output types.")