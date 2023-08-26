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

import argparse
import sys, os
from pathlib import Path

from nemo.deploy import DeployPyTriton, NemoQuery
from nemo.export import TensorRTLLM
from nemo.utils import logging

try:
    from contextlib import nullcontext
except ImportError:
    # handle python < 3.7
    from contextlib import suppress as nullcontext


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Deploy nemo models to Triton",
    )
    parser.add_argument("--nemo_checkpoint", type=str, help="Source .nemo file")
    parser.add_argument(
        "--model_type",
        type=str, default="gptnext",
        choices=["gptnext", "llama"],
        help="Type of the model. gpt or llama are only supported."
    )
    parser.add_argument("--triton_model_name", required=True, type=str, help="Name for the service")
    parser.add_argument("--triton_model_version", default=1, type=int, help="Version for the service")
    parser.add_argument("--triton_port", default=8000, type=int, help="Port for the Triton server to listen for requests")
    parser.add_argument("--triton_http_address", default="0.0.0.0", type=str, help="HTTP address for the Triton server")
    parser.add_argument("--trt_llm_folder", default=None, type=str, help="Folder for the trt-llm conversion")
    parser.add_argument("--num_gpus", default=1, type=int, help="Number of GPUs for the deployment")
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp8", "int8"],
        default="bf16",
        type=str,
        help="dtype of the model on TensorRT-LLM",
    )
    parser.add_argument("--max_input_len", default=200, type=int, help="Max input length of the model")
    parser.add_argument("--max_output_len", default=200, type=int, help="Max output length of the model")
    parser.add_argument("--max_batch_size", default=200, type=int, help="Max batch size of the model")

    args = parser.parse_args(argv)
    return args


def nemo_deploy(argv):
    args = get_args(argv)
    loglevel = logging.INFO
    logging.setLevel(loglevel)
    logging.info("Logging level set to {}".format(loglevel))
    logging.info(args)

    if args.dtype != "bf16":
        logging.error("Only bf16 is currently supported for the optimized deployment with TensorRT-LLM. "
                      "Support for the other precisions will be added in the coming releases.")
        return

    if args.trt_llm_folder is None:
        trt_llm_path = "/tmp/trt_llm_model_dir/"
        logging.info(
            "/tmp/trt_llm_model_dir/ path will be used as the TensorRT LLM folder. "
            "Please set this parameter if you'd like to use a path that has already "
            "included the TensorRT LLM model files."
        )
        Path(trt_llm_path).mkdir(parents=True, exist_ok=True)
    else:
        trt_llm_path = args.trt_llm_folder

    if args.nemo_checkpoint is None and not os.path.isdir(args.trt_llm_folder):
        logging.error(
            "The provided trt_llm_folder is not a valid TensorRT-LLM model "
            "directory. Please provide a --nemo_checkpoint."
        )

    trt_llm_exporter = TensorRTLLM(model_dir=trt_llm_path)

    if args.nemo_checkpoint is not None:
        trt_llm_exporter.export(
            nemo_checkpoint_path=args.nemo_checkpoint,
            model_type=args.model_type,
            n_gpus=args.num_gpus,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            max_batch_size=args.max_batch_size,
        )

    nm = DeployPyTriton(
        model=trt_llm_exporter,
        triton_model_name=args.triton_model_name,
        triton_model_version=args.triton_model_version,
        max_batch_size=args.max_batch_size,
        port=args.triton_port,
        http_address=args.triton_http_address,
    )
    
    nm.deploy()

    try:
        logging.info("Triton deploy function is called.")
        nm.serve()
    except:
        logging.info("An error has occurred and will stop serving the model.")

    logging.info("Model serving will be stopped.")
    nm.stop()


if __name__ == '__main__':
    nemo_deploy(sys.argv[1:])
