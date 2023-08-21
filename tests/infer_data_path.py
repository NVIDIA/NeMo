# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


def get_infer_test_data():
    test_data = {}

    test_data["GPT-843M-base"] = {}
    test_data["GPT-843M-base"]["model_type"] = "gptnext"
    test_data["GPT-843M-base"]["total_gpus"] = [1, 2]
    test_data["GPT-843M-base"]["location"] = "Selene"
    test_data["GPT-843M-base"]["trt_llm_model_dir"] = "/tmp/GPT-843M-base/trt_llm_model/"
    test_data["GPT-843M-base"][
        "checkpoint"
    ] = "/opt/checkpoints/GPT-843M-base/GPT-843M-1.nemo"

    test_data["GPT-2B-HF-base"] = {}
    test_data["GPT-2B-HF-base"]["model_type"] = "gptnext"
    test_data["GPT-2B-HF-base"]["total_gpus"] = [1, 2]
    test_data["GPT-2B-HF-base"]["location"] = "HF"
    test_data["GPT-2B-HF-base"]["trt_llm_model_dir"] = "/tmp/GPT-2B-hf-base/trt_llm_model/"
    test_data["GPT-2B-HF-base"]["checkpoint_dir"] = "/tmp/GPT-2B-hf-base/nemo_checkpoint/"
    test_data["GPT-2B-HF-base"]["checkpoint"] = (
        "/opt/checkpoints/GPT-2B.nemo"
    )
    test_data["GPT-2B-HF-base"]["checkpoint_link"] = (
        "https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo"
    )
    test_data["GPT-2B-HF-base"]["ptuned"] = {}
    test_data["GPT-2B-HF-base"]["ptuned"]["squad"] = "/opt/checkpoints/8b_squad_megatron_gpt_peft_tuning.nemo"

    test_data["GPT-2B-base"] = {}
    test_data["GPT-2B-base"]["model_type"] = "gptnext"
    test_data["GPT-2B-base"]["total_gpus"] = [1, 2]
    test_data["GPT-2B-base"]["location"] = "Selene"
    test_data["GPT-2B-base"]["trt_llm_model_dir"] = "/tmp/GPT-2B-base/trt_llm_model/"
    test_data["GPT-2B-base"]["checkpoint"] = "/opt/checkpoints/GPT-2B-base/megatron_converted_2b_tp1_pp1.nemo"

    test_data["GPT-8B-base"] = {}
    test_data["GPT-8B-base"]["model_type"] = "gptnext"
    test_data["GPT-8B-base"]["total_gpus"] = [1, 2]
    test_data["GPT-8B-base"]["location"] = "Selene"
    test_data["GPT-8B-base"]["trt_llm_model_dir"] = "/tmp/GPT-8B-base/trt_llm_model/"
    test_data["GPT-8B-base"]["checkpoint"] = "/opt/checkpoints/GPT-8B-base/megatron_converted_8b_tp4_pp1.nemo"

    test_data["GPT-43B-base"] = {}
    test_data["GPT-43B-base"]["model_type"] = "gptnext"
    test_data["GPT-43B-base"]["total_gpus"] = [1, 2]
    test_data["GPT-43B-base"]["location"] = "Selene"
    test_data["GPT-43B-base"]["trt_llm_model_dir"] = "/tmp/GPT-43B-base/trt_llm_model/"
    test_data["GPT-43B-base"][
        "checkpoint"
    ] = "/opt/checkpoints/GPT-43B-base/megatron_converted_43b_tp8_pp1.nemo"

    test_data["LLAMA2-7B-base"] = {}
    test_data["LLAMA2-7B-base"]["model_type"] = "llama"
    test_data["LLAMA2-7B-base"]["total_gpus"] = [1, 2]
    test_data["LLAMA2-7B-base"]["location"] = "Selene"
    test_data["LLAMA2-7B-base"]["trt_llm_model_dir"] = "/tmp/LLAMA2-7B-base/trt_llm_model/"
    test_data["LLAMA2-7B-base"][
        "checkpoint"
    ] = "/opt/checkpoints/LLAMA2-7B-base/llama2-7b.nemo"

    return test_data
