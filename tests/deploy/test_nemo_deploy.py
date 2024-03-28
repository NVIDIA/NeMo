# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import shutil
import time
import urllib.request as req
from pathlib import Path

import numpy as np
import pytest
import torch
from tqdm import tqdm

from nemo.deploy import DeployPyTriton, DeployTensorRTLLM, NemoQuery, NemoQueryTensorRTLLM
from nemo.export import TensorRTLLM
from tests.infer_data_path import download_nemo_checkpoint, get_infer_test_data


def get_accuracy_with_lambada(nq, use_async=False):
    # lambada dataset based accuracy test, which includes more than 5000 sentences.
    # Use generated last token with original text's last token for accuracy comparison.
    # If the generated last token start with the original token, trtllm_correct make an increment.
    # It generates a CSV file for text comparison detail.

    trtllm_correct = 0
    trtllm_correct_relaxed = 0
    all_expected_outputs = []
    all_trtllm_outputs = []

    with open('/opt/NeMo/tests/deploy/lambada.json', 'r') as file:
        records = json.load(file)

        if not use_async:
            for record in tqdm(records):
                prompt = record["text_before_last_word"]
                expected_output = record["last_word"].strip().lower()
                trtllm_output = nq.query_llm(prompts=[prompt], max_output_token=1, top_k=1, top_p=0.0, temperature=0.1)
                if isinstance(trtllm_output[0], str):
                    trtllm_output = trtllm_output[0].strip().lower()
                else:
                    trtllm_output = trtllm_output[0][0].strip().lower()
                all_expected_outputs.append(expected_output)
                all_trtllm_outputs.append(trtllm_output)
        else:
            prompts = []
            for record in tqdm(records):
                prompt = record["text_before_last_word"]
                prompts.append(prompt)
                expected_output = record["last_word"].strip().lower()
                all_expected_outputs.append(expected_output)
            all_trtllm_outputs = nq.query_llm(prompts=prompts, max_output_token=1, top_k=1, top_p=0.0, temperature=0.1)
            all_trtllm_outputs = [trtllm_output[0][0].strip().lower() for trtllm_output in all_trtllm_outputs]

        for trtllm_output, expected_output in zip(all_trtllm_outputs, all_expected_outputs):
            if expected_output == trtllm_output or trtllm_output.startswith(expected_output):
                trtllm_correct += 1

            if (
                expected_output == trtllm_output
                or trtllm_output.startswith(expected_output)
                or expected_output.startswith(trtllm_output)
            ):
                if len(trtllm_output) == 1 and len(expected_output) > 1:
                    continue
                trtllm_correct_relaxed += 1

            # print("-- expected_output: {0} and trtllm_output: {1}".format(expected_output, trtllm_output))

    trtllm_accuracy_relaxed = trtllm_correct_relaxed / len(all_expected_outputs)
    trtllm_accuracy = trtllm_correct / len(all_expected_outputs)
    return trtllm_accuracy, trtllm_accuracy_relaxed, all_trtllm_outputs, all_expected_outputs


def run_trt_llm_export(model_name, n_gpu, skip_accuracy=False, use_pytriton=True, use_streaming=False):
    test_data = get_infer_test_data()

    model_info = test_data[model_name]
    if model_info["location"] == "HF":
        download_nemo_checkpoint(model_info["checkpoint_link"], model_info["checkpoint_dir"], model_info["checkpoint"])

    print(
        "Path: {0} and model: {1} is next and test will start if the nemo checkpoint exists ...".format(
            model_info["checkpoint"], model_name
        )
    )
    if Path(model_info["checkpoint"]).exists():
        if n_gpu > torch.cuda.device_count():
            print(
                "Path: {0} and model: {1} with {2} gpus won't be tested since available # of gpus = {3}".format(
                    model_info["checkpoint"], model_name, n_gpu, torch.cuda.device_count()
                )
            )
            return

        Path(model_info["trt_llm_model_dir"]).mkdir(parents=True, exist_ok=True)

        print("")
        print("")
        print(
            "################################################## NEW TEST ##################################################"
        )
        print("")

        print(
            "Path: {0} and model: {1} with {2} gpus will be tested".format(model_info["checkpoint"], model_name, n_gpu)
        )
        trt_llm_exporter = TensorRTLLM(model_dir=model_info["trt_llm_model_dir"])
        use_inflight_batching = not use_pytriton

        trt_llm_exporter.export(
            nemo_checkpoint_path=model_info["checkpoint"],
            model_type=model_info["model_type"],
            n_gpus=n_gpu,
            max_input_token=1024,
            max_output_token=128,
            max_batch_size=model_info["max_batch_size"],
            use_inflight_batching=use_inflight_batching,
        )

        prompts = model_info["prompt_template"]
        if use_pytriton:
            if not use_streaming:
                nm = DeployPyTriton(model=trt_llm_exporter, triton_model_name=model_name, port=8000,)
                nm.deploy()
                nm.run()
                nq = NemoQuery(url="http://localhost:8000", model_name=model_name)
                output = nq.query_llm(
                    prompts=prompts,
                    max_output_token=model_info["max_output_token"],
                    top_k=1,
                    top_p=0.0,
                    temperature=1.0,
                )
            else:
                nm = DeployPyTriton(model=trt_llm_exporter, triton_model_name=model_name, port=8001, streaming=True,)

                nm.deploy()
                nm.run()
                nq = NemoQuery(url="grpc://localhost:8001", model_name=model_name)
                output_gen = nq.query_llm_streaming(
                    prompts=prompts,
                    max_output_token=model_info["max_output_token"],
                    top_k=1,
                    top_p=0.0,
                    temperature=1.0,
                )
                output = [cur_output for cur_output in output_gen][-1]
        else:
            model_repo_dir = "/tmp/ensemble"
            nm = DeployTensorRTLLM(
                model=trt_llm_exporter, triton_model_name=model_name, port=8000, model_repo_dir=model_repo_dir
            )
            nm.deploy()
            nm.run()
            time.sleep(20)
            nq = NemoQueryTensorRTLLM(url="localhost:8000", model_name="ensemble")
            output = nq.query_llm(
                prompts=prompts, max_output_token=model_info["max_output_token"], top_k=1, top_p=0.0, temperature=1.0,
            )

        print("")
        print("--- Prompt: ", prompts)
        print("")
        print("--- Output: ", output)
        print("")

        if not skip_accuracy:
            print("Start model accuracy testing ...")
            (
                trtllm_accuracy,
                trtllm_accuracy_relaxed,
                all_trtllm_outputs,
                all_expected_outputs,
            ) = get_accuracy_with_lambada(nq, use_async=False)
            print("Model Accuracy: {0}, Relaxed Model Accuracy: {1}".format(trtllm_accuracy, trtllm_accuracy_relaxed))
            assert trtllm_accuracy_relaxed > 0.5, "Model accuracy is below 0.5"

        trt_llm_exporter = None
        nm.stop()
        shutil.rmtree(model_info["trt_llm_model_dir"])


@pytest.mark.parametrize("n_gpus", [1])
def test_NV_GPT_8B_Base_4k_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""

    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Base-4k", n_gpus)


@pytest.mark.parametrize("n_gpus", [1])
def test_NV_GPT_8B_QA_4k_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-QA-4k", n_gpus)


@pytest.mark.parametrize("n_gpus", [1])
def test_NV_GPT_8B_Chat_4k_SFT_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Chat-4k-SFT", n_gpus)


@pytest.mark.parametrize("n_gpus", [1])
def test_NV_GPT_8B_Chat_4k_RLHF_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Chat-4k-RLHF", n_gpus)


@pytest.mark.parametrize("n_gpus", [1])
def test_NV_GPT_8B_Chat_4k_SteerLM_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Chat-4k-SteerLM", n_gpus)


@pytest.mark.parametrize("n_gpus", [1])
def test_LLAMA2_7B_base_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-7B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [1])
def test_LLAMA2_7B_base_1gpu_ifb(n_gpus):
    """Here we test the trt-llm transfer and infer function with IFB and c++ backend"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-7B-base", n_gpus, use_pytriton=False, skip_accuracy=False)


@pytest.mark.parametrize("n_gpus", [1])
def test_STARCODER1_15B_base_1gpu_ifb(n_gpus):
    """Here we test the trt-llm transfer and infer function with IFB and c++ backend"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("STARCODER1-15B-base", n_gpus, use_pytriton=False, skip_accuracy=False)


@pytest.mark.parametrize("n_gpus", [1])
def test_LLAMA2_7B_base_1gpu_streaming(n_gpus):
    """Here we test the trt-llm transfer and infer function with IFB and c++ backend"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-7B-base", n_gpus, use_pytriton=True, skip_accuracy=True, use_streaming=True)


@pytest.mark.parametrize("n_gpus", [1])
def test_LLAMA2_13B_base_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-13B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [2])
def test_LLAMA2_70B_base_2gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-70B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [1])
def test_GPT_2B_001_bf16_tp1_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("GPT-2B-001-bf16-tp1", n_gpus, skip_accuracy=True)
