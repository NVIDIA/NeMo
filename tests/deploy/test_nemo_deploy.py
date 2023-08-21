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
from nemo.export import TensorRTLLM
from tests.infer_data_path import get_infer_test_data


class TestNemoDeployment:
    @pytest.mark.unit
    @pytest.mark.skip()
    def test_in_framework_pytriton(self):
        """Here we test the in framework inference deployment to triton"""

        test_data = get_infer_test_data()
        test_at_least_one = False
        no_error = True

        for model_name, model_info in test_data.items():
            if model_info["location"] == "HF":
                self._download_nemo_checkpoint(
                    model_info["checkpoint_link"], model_info["checkpoint_dir"], model_info["checkpoint"]
                )

            if Path(model_info["checkpoint"]).exists():
                test_at_least_one = True
                nm = DeployPyTriton(checkpoint_path=model_info["checkpoint"], triton_model_name=model_name, port=8005)

                try:
                    nm.deploy()
                    nm.run()
                    nq = NemoQuery(url="localhost:8005", model_name=model_name)

                    output = nq.query_llm(prompts=["hello, testing GPT inference"])
                    print(output)
                except:
                    print("Couldn't start the model.")
                    no_error = False

                nm.stop()
            else:
                print("Model {0} could not be found at this location {1}".format(model_name, model_info["checkpoint"]))

        assert test_at_least_one, "At least one nemo checkpoint has to be tested."
        assert no_error, "At least one model couldn't be served successfully."

    @pytest.mark.unit
    def test_trt_llm_pytriton(self):
        """Here we test the in framework inference deployment to triton"""

        test_data = get_infer_test_data()
        test_at_least_one = False
        no_error = True

        for model_name, model_info in test_data.items():
            if model_info["location"] == "HF":
                self._download_nemo_checkpoint(
                    model_info["checkpoint_link"], model_info["checkpoint_dir"], model_info["checkpoint"]
                )

            if Path(model_info["checkpoint"]).exists():
                test_at_least_one = True
                Path(model_info["trt_llm_model_dir"]).mkdir(parents=True, exist_ok=True)
                trt_llm_exporter = TensorRTLLM(model_dir=model_info["trt_llm_model_dir"])
                trt_llm_exporter.export(nemo_checkpoint_path=model_info["checkpoint"], n_gpus=1)

                nm = DeployPyTriton(model=trt_llm_exporter, triton_model_name=model_name, port=8000)

                try:
                    nm.deploy()
                    nm.run()
                    nq = NemoQuery(url="localhost", model_name=model_name)

                    output = nq.query_llm(prompts=["hello, testing GPT inference", "another GPT inference test?"])
                    print(output)
                except:
                    print("Couldn't start the model.")
                    no_error = False

                nm.stop()
            else:
                print("Model {0} could not be found at this location {1}".format(model_name, model_info["checkpoint"]))

        assert test_at_least_one, "At least one nemo checkpoint has to be tested."
        assert no_error, "At least one model couldn't be served successfully."

    def _prep_test_data(self):
        self.test_data = {}

        self.test_data["GPT-843M-base"] = {}
        self.test_data["GPT-843M-base"]["location"] = "Selene"
        self.test_data["GPT-843M-base"]["trt_llm_model_dir"] = "/tmp/GPT-843M-base/trt_llm_model/"
        self.test_data["GPT-843M-base"][
            "checkpoint"
        ] = "/opt/checkpoints/GPT-843M-base/megatron_converted_843m_tp1_pp1.nemo"

        self.test_data["GPT-2B-HF-base"] = {}
        self.test_data["GPT-2B-HF-base"]["location"] = "HF"
        self.test_data["GPT-2B-HF-base"]["trt_llm_model_dir"] = "/tmp/GPT-2B-hf-base/trt_llm_model/"
        self.test_data["GPT-2B-HF-base"]["checkpoint_dir"] = "/tmp/GPT-2B-hf-base/nemo_checkpoint/"
        self.test_data["GPT-2B-HF-base"]["checkpoint"] = (
            self.test_data["GPT-2B-HF-base"]["checkpoint_dir"] + "GPT-2B-001_bf16_tp1.nemo"
        )
        self.test_data["GPT-2B-HF-base"]["checkpoint_link"] = (
            "https://huggingface.co/nvidia/GPT-2B-001/resolve/main/" "GPT-2B-001_bf16_tp1.nemo"
        )

        self.test_data["GPT-2B-base"] = {}
        self.test_data["GPT-2B-base"]["location"] = "Selene"
        self.test_data["GPT-2B-base"]["trt_llm_model_dir"] = "/tmp/GPT-2B-base/trt_llm_model/"
        self.test_data["GPT-2B-base"]["checkpoint"] = "/opt/checkpoints/GPT-2B-base/megatron_converted_2b_tp1_pp1.nemo"

        self.test_data["GPT-8B-base"] = {}
        self.test_data["GPT-8B-base"]["location"] = "Selene"
        self.test_data["GPT-8B-base"]["trt_llm_model_dir"] = "/tmp/GPT-8B-base/trt_llm_model/"
        self.test_data["GPT-8B-base"]["checkpoint"] = "/opt/checkpoints/GPT-8B-base/megatron_converted_8b_tp4_pp1.nemo"

        self.test_data["GPT-43B-base"] = {}
        self.test_data["GPT-43B-base"]["location"] = "Selene"
        self.test_data["GPT-43B-base"]["trt_llm_model_dir"] = "/tmp/GPT-43B-base/trt_llm_model/"
        self.test_data["GPT-43B-base"][
            "checkpoint"
        ] = "/opt/checkpoints/GPT-43B-base/megatron_converted_43b_tp8_pp1.nemo"

    def _download_nemo_checkpoint(self, checkpoint_link, checkpoint_dir, checkpoint_path):
        if not Path(checkpoint_path).exists():
            print("Checkpoint: {0}, will be downloaded to {1}".format(checkpoint_link, checkpoint_path))
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            req.urlretrieve(checkpoint_link, checkpoint_path)
            print("Checkpoint: {0}, download completed.".format(checkpoint_link))
        else:
            print("Checkpoint: {0}, has already been downloaded.".format(checkpoint_link))
