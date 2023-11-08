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


def get_infer_test_data():
    test_data = {}

    '''
    test_data["GPT-2B-HF-base"] = {}
    test_data["GPT-2B-HF-base"]["model_type"] = "gptnext"
    test_data["GPT-2B-HF-base"]["total_gpus"] = [1]
    test_data["GPT-2B-HF-base"]["location"] = "HF"
    test_data["GPT-2B-HF-base"]["trt_llm_model_dir"] = "/tmp/GPT-2B-hf-base/trt_llm_model-1/"
    test_data["GPT-2B-HF-base"]["checkpoint_dir"] = "/tmp/GPT-2B-hf-base/nemo_checkpoint/"
    test_data["GPT-2B-HF-base"]["checkpoint"] = (
        "/opt/checkpoints/GPT-2B.nemo"
    )
    test_data["GPT-2B-HF-base"]["checkpoint_link"] = (
        "https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo"
    )
    '''

    test_data["NV-GPT-8B-Base-4k"] = {}
    test_data["NV-GPT-8B-Base-4k"]["model_type"] = "gptnext"
    test_data["NV-GPT-8B-Base-4k"]["total_gpus"] = [1, 2, 4, 8]
    test_data["NV-GPT-8B-Base-4k"]["location"] = "Local"
    test_data["NV-GPT-8B-Base-4k"]["trt_llm_model_dir"] = "/tmp/NV-GPT-8B-Base-4k/nv-gpt-8b-base-4k_v1.0/"
    test_data["NV-GPT-8B-Base-4k"]["checkpoint"] = "/opt/checkpoints/NV-GPT-8B-Base-4k/nv-gpt-8b-base-4k_v1.0/NV-GPT-8B-Base-4k.nemo"

    test_data["NV-GPT-8B-Base-16k"] = {}
    test_data["NV-GPT-8B-Base-16k"]["model_type"] = "gptnext"
    test_data["NV-GPT-8B-Base-16k"]["total_gpus"] = [1, 2, 4, 8]
    test_data["NV-GPT-8B-Base-16k"]["location"] = "Local"
    test_data["NV-GPT-8B-Base-16k"]["trt_llm_model_dir"] = "/tmp/NV-GPT-8B-Base-16k/nv-gpt-8b-base-16k_v1.0/"
    test_data["NV-GPT-8B-Base-16k"]["checkpoint"] = "/opt/checkpoints/NV-GPT-8B-Base-16k/nv-gpt-8b-base-16k_v1.0/NV-GPT-8B-Base-16k.nemo"

    test_data["NV-GPT-8B-QA-4k"] = {}
    test_data["NV-GPT-8B-QA-4k"]["model_type"] = "gptnext"
    test_data["NV-GPT-8B-QA-4k"]["total_gpus"] = [1, 2, 4, 8]
    test_data["NV-GPT-8B-QA-4k"]["location"] = "Local"
    test_data["NV-GPT-8B-QA-4k"]["trt_llm_model_dir"] = "/tmp/NV-GPT-8B-QA-4k/nv-gpt-8b-qa-4k_v1.0/"
    test_data["NV-GPT-8B-QA-4k"]["checkpoint"] = "/opt/checkpoints/NV-GPT-8B-QA-4k/nv-gpt-8b-qa-4k_v1.0/NV-GPT-8B-QA-4k.nemo"

    test_data["NV-GPT-8B-Chat-4k-SFT"] = {}
    test_data["NV-GPT-8B-Chat-4k-SFT"]["model_type"] = "gptnext"
    test_data["NV-GPT-8B-Chat-4k-SFT"]["total_gpus"] = [1, 2, 4, 8]
    test_data["NV-GPT-8B-Chat-4k-SFT"]["location"] = "Local"
    test_data["NV-GPT-8B-Chat-4k-SFT"]["trt_llm_model_dir"] = "/tmp/NV-GPT-8B-Chat-4k-SFT/nv-gpt-8b-chat-4k-sft_v1.0/"
    test_data["NV-GPT-8B-Chat-4k-SFT"]["checkpoint"] = "/opt/checkpoints/NV-GPT-8B-Chat-4k-SFT/nv-gpt-8b-chat-4k-sft_v1.0/NV-GPT-8B-Chat-4k-SFT.nemo"

    test_data["NV-GPT-8B-Chat-4k-RLHF"] = {}
    test_data["NV-GPT-8B-Chat-4k-RLHF"]["model_type"] = "gptnext"
    test_data["NV-GPT-8B-Chat-4k-RLHF"]["total_gpus"] = [1, 2, 4, 8]
    test_data["NV-GPT-8B-Chat-4k-RLHF"]["location"] = "Local"
    test_data["NV-GPT-8B-Chat-4k-RLHF"]["trt_llm_model_dir"] = "/tmp/NV-GPT-8B-Chat-4k-RLHF/nv-gpt-8b-chat-4k-rlhf_v1.0/"
    test_data["NV-GPT-8B-Chat-4k-RLHF"]["checkpoint"] = "/opt/checkpoints/NV-GPT-8B-Chat-4k-RLHF/nv-gpt-8b-chat-4k-rlhf_v1.0/NV-GPT-8B-Chat-4k-RLHF.nemo"

    test_data["NV-GPT-8B-Chat-4k-SteerLM"] = {}
    test_data["NV-GPT-8B-Chat-4k-SteerLM"]["model_type"] = "gptnext"
    test_data["NV-GPT-8B-Chat-4k-SteerLM"]["total_gpus"] = [1, 2, 4, 8]
    test_data["NV-GPT-8B-Chat-4k-SteerLM"]["location"] = "Local"
    test_data["NV-GPT-8B-Chat-4k-SteerLM"]["trt_llm_model_dir"] = "/tmp/NV-GPT-8B-Chat-4k-SteerLM/nv-gpt-8b-chat-4k-steerlm_v1.0/"
    test_data["NV-GPT-8B-Chat-4k-SteerLM"]["checkpoint"] = "/opt/checkpoints/NV-GPT-8B-Chat-4k-SteerLM/nv-gpt-8b-chat-4k-steerlm_v1.0/NV-GPT-8B-Chat-4k-SteerLM.nemo"

    test_data["LLAMA2-7B-base"] = {}
    test_data["LLAMA2-7B-base"]["model_type"] = "llama"
    test_data["LLAMA2-7B-base"]["total_gpus"] = [1, 2, 4, 8]
    test_data["LLAMA2-7B-base"]["location"] = "Local"
    test_data["LLAMA2-7B-base"]["trt_llm_model_dir"] = "/tmp/LLAMA2-7B-base/trt_llm_model-1/"
    test_data["LLAMA2-7B-base"]["checkpoint"] = "/opt/checkpoints/LLAMA2-7B-base/LLAMA2-7B-base-1.nemo"
    # test_data["LLAMA2-7B-base"]["p_tuning_checkpoint"] = "/opt/checkpoints/LLAMA2-7B-PTuning/LLAMA2-7B-PTuning-1.nemo"

    test_data["LLAMA2-13B-base"] = {}
    test_data["LLAMA2-13B-base"]["model_type"] = "llama"
    test_data["LLAMA2-13B-base"]["total_gpus"] = [1, 2, 4, 8]
    test_data["LLAMA2-13B-base"]["location"] = "Local"
    test_data["LLAMA2-13B-base"]["trt_llm_model_dir"] = "/tmp/LLAMA2-13B-base/trt_llm_model-1/"
    test_data["LLAMA2-13B-base"]["checkpoint"] = "/opt/checkpoints/LLAMA2-13B-base/LLAMA2-13B-base-1.nemo"
    test_data["LLAMA2-13B-base"]["p_tuning_checkpoint"] = "/opt/checkpoints/LLAMA2-13B-PTuning/LLAMA2-13B-PTuning-1.nemo"

    test_data["LLAMA2-70B-base"] = {}
    test_data["LLAMA2-70B-base"]["model_type"] = "llama"
    test_data["LLAMA2-70B-base"]["total_gpus"] = [2, 4, 8]
    test_data["LLAMA2-70B-base"]["location"] = "Local"
    test_data["LLAMA2-70B-base"]["trt_llm_model_dir"] = "/tmp/LLAMA2-70B-base/trt_llm_model-1/"
    test_data["LLAMA2-70B-base"]["checkpoint"] = "/opt/checkpoints/LLAMA2-70B-base/LLAMA2-70B-base-1.nemo"

    test_data["FALCON-7B-base"] = {}
    test_data["FALCON-7B-base"]["model_type"] = "falcon"
    test_data["FALCON-7B-base"]["total_gpus"] = [1, 2, 4, 8]
    test_data["FALCON-7B-base"]["location"] = "Local"
    test_data["FALCON-7B-base"]["trt_llm_model_dir"] = "/tmp/FALCON-7B-base/trt_llm_model-1/"
    test_data["FALCON-7B-base"]["checkpoint"] = "/opt/checkpoints/FALCON-7B-base/FALCON-7B-base-1.nemo"

    test_data["FALCON-40B-base"] = {}
    test_data["FALCON-40B-base"]["model_type"] = "falcon"
    test_data["FALCON-40B-base"]["total_gpus"] = [2, 4, 8]
    test_data["FALCON-40B-base"]["location"] = "Local"
    test_data["FALCON-40B-base"]["trt_llm_model_dir"] = "/tmp/FALCON-40B-base/trt_llm_model-1/"
    test_data["FALCON-40B-base"]["checkpoint"] = "/opt/checkpoints/FALCON-40B-base/FALCON-40B-base-1.nemo"

    return test_data


def download_nemo_checkpoint(checkpoint_link, checkpoint_dir, checkpoint_path):
    if not Path(checkpoint_path).exists():
        print("Checkpoint: {0}, will be downloaded to {1}".format(checkpoint_link, checkpoint_path))
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        ckp_path = Path("/opt/checkpoints/")
        if not ckp_path.exists():
            ckp_path.mkdir(parents=True, exist_ok=False)
        req.urlretrieve(checkpoint_link, checkpoint_path)
        print("Checkpoint: {0}, download completed.".format(checkpoint_link))
    else:
        print("Checkpoint: {0}, has already been downloaded.".format(checkpoint_link))
