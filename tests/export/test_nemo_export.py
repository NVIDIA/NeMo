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
from tests.infer_data_path import get_infer_test_data, download_nemo_checkpoint


class TestNemoExport:
    @pytest.mark.unit
    def test_trt_llm_export(self):
        """Here we test the trt-llm transfer and infer function"""

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
                for n_gpu in model_info["total_gpus"]:
                    print(
                        "Path: {0} and model: {1} with {2} gpus will be tested".format(
                            model_info["checkpoint"], model_name, n_gpu
                        )
                    )
                    Path(model_info["trt_llm_model_dir"]).mkdir(parents=True, exist_ok=True)
                    trt_llm_exporter = TensorRTLLM(model_dir=model_info["trt_llm_model_dir"])
                    trt_llm_exporter.export(
                        nemo_checkpoint_path=model_info["checkpoint"],
                        model_type=model_info["model_type"],
                        n_gpus=n_gpu,
                    )
                    output = trt_llm_exporter.forward(["Hi, how are you?", "I am good, thanks, how about you?"])
                    print("output 1: ", output)

                    trt_llm_exporter2 = TensorRTLLM(model_dir=model_info["trt_llm_model_dir"])
                    output = trt_llm_exporter2.forward(["Let's see how this works", "Did you get the result yet?"])
                    print("output 2: ", output)

                test_at_least_one = True

        assert test_at_least_one, "At least one nemo checkpoint has to be tested."

    @pytest.mark.skip()
    @pytest.mark.unit
    def test_trt_llm_export_ptuned(self):
        """Here we test the trt-llm transfer and infer function"""

        test_data = get_infer_test_data()

        test_at_least_one = False

        for model_name, model_info in test_data.items():
            if model_info["location"] == "HF":
                download_nemo_checkpoint(
                    model_info["checkpoint_link"], model_info["checkpoint_dir"], model_info["checkpoint"]
                )

            if Path(model_info["checkpoint"]).exists():
                if "ptuned" in model_info:
                    for task, path in model_info["ptuned"].items():
                        for n_gpu in model_info["total_gpus"]:
                            print(
                                "Path: {0}, Task{1} and model: {2} with {3} gpus will be tested".format(
                                    path, task, model_name, n_gpu
                                )
                            )
                            Path(model_info["trt_llm_model_dir"]).mkdir(parents=True, exist_ok=True)
                            trt_llm_exporter = TensorRTLLM(model_dir=model_info["trt_llm_model_dir"])
                            trt_llm_exporter.export(
                                nemo_checkpoint_path=model_info["checkpoint"],
                                prompt_checkpoint_path=path,
                                n_gpus=n_gpu,
                            )
                            output = trt_llm_exporter.forward(["test1", "how about test 2"])
                            print("output 1: ", output)

                    test_at_least_one = True

        assert test_at_least_one, "At least one nemo checkpoint has to be tested."

