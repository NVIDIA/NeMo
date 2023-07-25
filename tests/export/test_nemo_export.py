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

from nemo.export import TensorRTLLM


class TestNemoExport:

    nemo_checkpoint_link = "https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo"
    nemo_checkpoint_path = "/opt/checkpoints/GPT-2B.nemo"
    trt_llm_model_dir = "/opt/checkpoints/GPT-2B-TRT-LLM/"

    @pytest.mark.unit
    def test_trt_llm_export(self):
        """Here we test the trt-llm export and infer function"""

        if not Path(self.nemo_checkpoint_path).exists():
            print("File will be downloaded...")
            req.urlretrieve(self.nemo_checkpoint_link, self.nemo_checkpoint_path)
            print("File download completed.")
        else:
            print("Checkpoint has already been downloaded.")

        trt_llm_exporter = TensorRTLLM(model_dir=self.trt_llm_model_dir)
        trt_llm_exporter.export(nemo_checkpoint_path=self.nemo_checkpoint_path, n_gpus=1)
        output = trt_llm_exporter.forward(["test1", "how about test 2"])
        print(output)
