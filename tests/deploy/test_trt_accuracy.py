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
import csv
import json

from nemo.deploy import DeployPyTriton, NemoQuery
from nemo.export import TensorRTLLM
from tests.infer_data_path import get_infer_test_data, download_nemo_checkpoint


class TestNemoDeployment:

    def _run_lambada_test(self, nq):
        #lambada dataset based accuracy test, which includes more than 5000 sentences.
        #Use generated last token with original text's last token for accuracy comparison.
        #If the generated last token start with the original token, trtllm_correct make an increment.
        #It generates a CSV file for text comparison detail.

        trtllm_correct = 0
        total_sentences = 0

        with open('/opt/NeMo/tests/deploy/lambada.json', 'r') as file:
            records = json.load(file)

            for record in records:
                total_sentences = total_sentences + 1
                lambada_input=[]

                lambada_input.append(record["text_before_last_word"])

                lambada_last_word = record["last_word"]

                trtllm_output = nq.query_llm(prompts=lambada_input, max_output_token=1, top_k=1, top_p=0, temperature=0.1)
                # print("---- trtllm output-----------")
                # print(trtllm_output)

                with open("eval_generate_token_result.csv", "a", encoding="UTF8") as f:
                    writer = csv.writer(f)

                    input_text = record["text"]
                    input_text_withno_lasttoken = lambada_input[0]

                    trtllm_generated = trtllm_output[0][0].lstrip()

                    # print("------expect last word")
                    # print(lambada_last_word)
                    # print("------trt llm output------")
                    # print(trtllm_generated)

                    if trtllm_generated.startswith(lambada_last_word):
                        trtllm_correct = trtllm_correct + 1

                    writer.writerow([input_text,input_text_withno_lasttoken,lambada_last_word,trtllm_generated])

        print("----- TRT-LLM accuracy-----")
        trtllm_accuracy = trtllm_correct / total_sentences
        print(trtllm_accuracy)

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

                self._run_lambada_test(nq)
                nm.stop()

                if "p_tuning_checkpoint" in model_info.keys() and Path(model_info["p_tuning_checkpoint"]).exists():
                    print(
                        "Path: {0} and model: {1} with {2} gpus will be tested with PTuning checkpoint {3}".format(
                            model_info["checkpoint"], model_name, n_gpu, model_info["p_tuning_checkpoint"]
                        )
                    )

                    trt_llm_exporter.export(
                        nemo_checkpoint_path=model_info["checkpoint"],
                        model_type=model_info["model_type"],
                        n_gpus=n_gpu,
                        prompt_embeddings_checkpoint_path=model_info["p_tuning_checkpoint"],
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
                    print("output with export using ptuning: ", output)
                    print("")

                    self._run_lambada_test(nq)
                    nm.stop()

            else:
                print("Model {0} could not be found at this location {1}".format(model_name, model_info["checkpoint"]))

        assert test_at_least_one, "At least one nemo checkpoint has to be tested."

