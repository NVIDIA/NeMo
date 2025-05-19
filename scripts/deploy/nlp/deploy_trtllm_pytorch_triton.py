# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import logging
import sys
import torch

from nemo.deploy import DeployPyTriton
from nemo.deploy.nlp.trtllm_pytorch_deployable import TensorRTLLMPyotrchDeployable

LOGGER = logging.getLogger("NeMo")


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Deploy TensorRT-LLM PyTorch models to Triton",
    )
    parser.add_argument(
        "-hp", "--hf_model_id_path", required=True, type=str, help="Path to the HuggingFace model or model identifier"
    )
    parser.add_argument("-t", "--tokenizer", type=str, help="Path to the tokenizer or tokenizer instance")
    parser.add_argument("-tmn", "--triton_model_name", required=True, type=str, help="Name for the service")
    parser.add_argument("-tmv", "--triton_model_version", default=1, type=int, help="Version for the service")
    parser.add_argument(
        "-trp", "--triton_port", default=8000, type=int, help="Port for the Triton server to listen for requests"
    )
    parser.add_argument(
        "-tha", "--triton_http_address", default="0.0.0.0", type=str, help="HTTP address for the Triton server"
    )
    parser.add_argument("-tps", "--tensor_parallel_size", default=1, type=int, help="Tensor parallelism size")
    parser.add_argument("-pps", "--pipeline_parallel_size", default=1, type=int, help="Pipeline parallelism size")
    parser.add_argument(
        "-meps", "--moe_expert_parallel_size", default=-1, type=int, help="MOE expert parallelism size"
    )
    parser.add_argument(
        "-mtps", "--moe_tensor_parallel_size", default=-1, type=int, help="MOE tensor parallelism size"
    )
    parser.add_argument("-mbs", "--max_batch_size", default=8, type=int, help="Max batch size of the model")
    parser.add_argument(
        "-mnt", "--max_num_tokens", default=8192, type=int, help="Maximum total tokens across all sequences in a batch"
    )
    parser.add_argument("-dt", "--dtype", default="auto", type=str, help="Model data type")
    parser.add_argument("-ab", "--attn_backend", default="TRTLLM", type=str, help="Attention kernel backend")
    parser.add_argument("-eos", "--enable_overlap_scheduler", action="store_true", help="Enable overlap scheduler")
    parser.add_argument("-ecp", "--enable_chunked_prefill", action="store_true", help="Enable chunked prefill")
    parser.add_argument("-ucg", "--use_cuda_graph", action="store_true", help="Use CUDA graph")
    parser.add_argument("-dm", "--debug_mode", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    return args


def trtllm_deploy():
    args = get_args()

    if args.debug_mode:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    LOGGER.setLevel(loglevel)
    LOGGER.info("Logging level set to {}".format(loglevel))
    LOGGER.info(args)

    model = TensorRTLLMPyotrchDeployable(
        hf_model_id_path=args.hf_model_id_path,
        tokenizer=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        moe_expert_parallel_size=args.moe_expert_parallel_size,
        moe_tensor_parallel_size=args.moe_tensor_parallel_size,
        max_batch_size=args.max_batch_size,
        max_num_tokens=args.max_num_tokens,
        dtype=args.dtype,
        attn_backend=args.attn_backend,
        enable_overlap_scheduler=args.enable_overlap_scheduler,
        enable_chunked_prefill=args.enable_chunked_prefill,
        use_cuda_graph=args.use_cuda_graph,
    )

    try:
        nm = DeployPyTriton(
            model=model,
            triton_model_name=args.triton_model_name,
            triton_model_version=args.triton_model_version,
            max_batch_size=args.max_batch_size,
            http_port=args.triton_port,
            address=args.triton_http_address,
        )

        LOGGER.info("Triton deploy function will be called.")
        nm.deploy()
    except Exception as error:
        LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))
        return

    try:
        LOGGER.info("Model serving on Triton will be started.")
        nm.serve()
    except Exception as error:
        LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))
        return

    LOGGER.info("Model serving will be stopped.")
    nm.stop()


if __name__ == '__main__':
    trtllm_deploy()
