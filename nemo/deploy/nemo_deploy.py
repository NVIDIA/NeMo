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
import importlib

from pytorch_lightning import Trainer
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton


class NemoDeploy:

    def __init__(self,
                 checkpoint_path: str,
                 triton_model_name: str,
                 inference_type: str="Normal",
                 model_name: str = "GPT",
                 model_type: str="LLM",
                 max_batch_size: int=128,
                 temp_nemo_dir=None,
    ):
        self.checkpoint_path = checkpoint_path
        self.triton_model_name = triton_model_name
        self.inference_type = inference_type
        self.model_name = model_name
        self.model_type = model_type
        self.max_batch_size = max_batch_size
        self.temp_nemo_dir = temp_nemo_dir
        self.model = None
        self.triton = None

        if self.temp_nemo_dir is None:
            print("write later")

    def deploy(self):
        self._init_nemo_model()
        # Connecting inference callable with Triton Inference Server
        try:
            self.triton = Triton()
            self.triton.bind(
                model_name=self.model_name,
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

    def _get_module_and_class(self, target: str):
        l = target.rindex(".")
        return target[0:l], target[l+1:len(target)]

    def _init_nemo_model(self):
        model_config = ModelPT.restore_from(self.checkpoint_path, return_config=True)
        module_path, class_name = self._get_module_and_class(model_config.target)
        cls = getattr(importlib.import_module(module_path), class_name)
        self.model = cls.restore_from(restore_path=self.checkpoint_path, trainer=Trainer())