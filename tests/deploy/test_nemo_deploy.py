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

import urllib.request as req
from pathlib import Path

import pytest

from nemo.deploy import DeployPyTriton, NemoQuery
from nemo.transfer import TensorRTLLM


class TestNemoDeployment:

    nemo_checkpoint_link = "https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo"
    nemo_checkpoint_path = "/opt/checkpoints/GPT-2B.nemo"
    temp_nemo_dir = "/opt/checkpoints/GPT-2B/"
    trt_llm_model_dir = "/opt/checkpoints/GPT-2B-TRT-LLM/"

    def __init__(self):
        self.test_data = {}

        self.test_data["GPT-846M-base"] = {}
        self.test_data["GPT-846M-base"]["checkpoint_location"] = "Selene"
        self.test_data["GPT-846M-base"]["checkpoint_path"] = ""
        self.test_data["GPT-846M-base"]["temp_nemo_dir"] = "/temp/GPT-846M-base/"
        self.test_data["GPT-846M-base"]["trt_llm_model_dir"] = "/temp/GPT-846M-base/"

    def _download_nemo_checkpoint(self):
        if not Path(self.nemo_checkpoint_path).exists():
            print("File will be downloaded...")
            req.urlretrieve(self.nemo_checkpoint_link, self.nemo_checkpoint_path)
            print("File download completed.")
        else:
            print("Checkpoint has already been downloaded.")

    @pytest.mark.unit
    def test_in_framework_pytriton(self):
        """Here we test the in framework inference deployment to triton"""
        triton_model_name = "GPT_2B"
        self._download_nemo_checkpoint()

        nm = DeployPyTriton(checkpoint_path=self.nemo_checkpoint_path, triton_model_name=triton_model_name)

        nm.deploy()
        nm.run()
        nq = NemoQuery(url="localhost", model_name=triton_model_name)

        output = nq.query_gpt(prompts=["hello, testing GPT inference"])
        print(output)

        nm.stop()

    @pytest.mark.unit
    def test_trt_llm_pytriton(self):
        """Here we test the optimized trt-llm inference deployment to triton"""
        triton_model_name = "GPT_2B_TRT_LLM"
        self._download_nemo_checkpoint()

        trt_llm_exporter = TensorRTLLM(model_dir=self.trt_llm_model_dir)
        trt_llm_exporter.transfer(nemo_checkpoint_path=self.nemo_checkpoint_path, n_gpus=1)

        nm = DeployPyTriton(model=trt_llm_exporter, triton_model_name=triton_model_name)

        nm.deploy()
        nm.run()
        nq = NemoQuery(url="localhost", model_name=triton_model_name)

        output = nq.query_gpt(prompts=["hello, testing GPT inference", "another GPT inference test?"])
        print(output)

        nm.stop()
