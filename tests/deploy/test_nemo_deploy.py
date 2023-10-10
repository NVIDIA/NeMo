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
import numpy as np

from nemo.deploy import DeployPyTriton, NemoQuery
from nemo.export import TensorRTLLM
from tests.infer_data_path import get_infer_test_data, download_nemo_checkpoint


class TestNemoDeployment:

    @pytest.mark.unit
    def test_trt_llm_pytriton(self):
        """Here we test the in framework inference deployment to triton"""

        test_data = get_infer_test_data()
        test_at_least_one = False

        for model_name, model_info in test_data.items():
            if model_info["location"] == "HF":
                download_nemo_checkpoint(
                    model_info["checkpoint_link"], model_info["checkpoint_dir"], model_info["checkpoint"]
                )

            print(
                "Path: {0} and model: {1} is next ...".format(
                    model_info["checkpoint"], model_name
                )
            )
            if Path(model_info["checkpoint"]).exists():
                n_gpu = model_info["total_gpus"][0]
                print(
                    "Path: {0} and model: {1} with {2} gpus will be tested".format(
                        model_info["checkpoint"], model_name, n_gpu
                    )
                )

                test_at_least_one = True
                Path(model_info["trt_llm_model_dir"]).mkdir(parents=True, exist_ok=True)

                trt_llm_exporter = TensorRTLLM(model_dir=model_info["trt_llm_model_dir"])
                trt_llm_exporter.export(
                    nemo_checkpoint_path=model_info["checkpoint"],
                    model_type=model_info["model_type"],
                    n_gpus=n_gpu,
                )

                nm = DeployPyTriton(model=trt_llm_exporter, triton_model_name=model_name, port=8000)
                nm.deploy()
                nm.run()
                nq = NemoQuery(url="http://localhost", model_name=model_name)

                prompts = ["hello, testing GPT inference", "another GPT inference test?"]
                output = nq.query_llm(
                    prompts=prompts,
                    max_output_token=200,
                )
                print("prompts: ", prompts)
                print("")
                print("output: ", output)
                print("")

                prompts = ["Give me some info about Paris", "Do you think Londan is a good city to visit?", "What do you think about Rome?"]
                output = nq.query_llm(
                    prompts=prompts,
                    max_output_token=200,
                )
                print("prompts: ", prompts)
                print("")
                print("output: ", output)
                print("")

                nm.stop()
            else:
                print("Model {0} could not be found at this location {1}".format(model_name, model_info["checkpoint"]))

        assert test_at_least_one, "At least one nemo checkpoint has to be tested."


