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

# from nemo.export import unpack_nemo_file


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

        if self.temp_nemo_dir is None:
            print("write later")

        self._init_nemo_model()

    def _init_nemo_model(self):
        model_config = ModelPT.restore_from(self.checkpoint_path, return_config=True)
        print(model_config.target)
        #cls = importlib.import_module(model_config.target)
        #self.model = ModelPT.restore_from(cls, self.checkpoint_path)

        #import nemo.collections.nlp.models.language_modeling.megatron_gpt_model.MegatronGPTModel as gpt
        #cls = gpt.restore_from(self.checkpoint_path)
        #print(cls)

    @batch
    def _infer_fn(self, **inputs: np.ndarray):
        (input1_batch,) = inputs.values()
        input1_batch_tensor = torch.from_numpy(input1_batch).to("cuda")
        output1_batch_tensor = self.model(input1_batch_tensor)  # Calling the Python model inference
        output1_batch = output1_batch_tensor.cpu().detach().numpy()
        return [output1_batch]

    def serve(self):
        return True