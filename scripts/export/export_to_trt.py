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
        description=f"Exports nemo models stored in nemo checkpoints to TensorRT-LLM",
    )

    parser.add_argument(
        "-nc",
        "--nemo_checkpoint",
        required=True,
        type=str,
        help="Source .nemo file"
    )

    parser.add_argument(
        "-pnc",
        "--ptuning_nemo_checkpoint",
        type=str,
        help="Source .nemo file for prompt embeddings table")

    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        required=True,
        choices=["gptnext", "llama"],
        help="Type of the model. gpt or llama are only supported."
    )

    parser.add_argument(
        "-mr",
        "--model_repository",
        required=True,
        default=None,
        type=str,
        help="Folder for the trt-llm model files"
    )

    parser.add_argument(
        "-ng",
        "--num_gpus",
        default=1,
        type=int,
        help="Number of GPUs for the deployment"
    )

    parser.add_argument(
        "-dt",
        "--dtype",
        choices=["bf16", "fp16", "fp8", "int8"],
        default="bf16",
        type=str,
        help="dtype of the model on TensorRT-LLM",
    )

    parser.add_argument(
        "-mil",
        "--max_input_len",
        default=200,
        type=int,
        help="Max input length of the model"
    )

    parser.add_argument(
        "-mol",
        "--max_output_len",
        default=200,
        type=int,
        help="Max output length of the model"
    )

    parser.add_argument(
        "-mbs",
        "--max_batch_size",
        default=200,
        type=int,
        help="Max batch size of the model"
    )

    parser.add_argument(
        "-dm",
        "--debug_mode",
        default="False",
        type=str,
        help="Enable debug mode"
    )

    args = parser.parse_args(argv)
    return args


def nemo_export(argv):
    args = get_args(argv)

    if args.debug_mode == "True":
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    logging.setLevel(loglevel)
    logging.info("Logging level set to {}".format(loglevel))
    logging.info(args)

    if args.dtype != "bf16":
        logging.error("Only bf16 is currently supported for the optimized deployment with TensorRT-LLM. "
                      "Support for the other precisions will be added in the coming releases.")
        return

    try:
        trt_llm_exporter = TensorRTLLM(model_dir=args.model_repository)

        logging.info("Export to TensorRT-LLM function is called.")
        trt_llm_exporter.export(
            nemo_checkpoint_path=args.nemo_checkpoint,
            model_type=args.model_type,
            prompt_embeddings_checkpoint_path=args.ptuning_nemo_checkpoint,
            n_gpus=args.num_gpus,
            max_input_token=args.max_input_len,
            max_output_token=args.max_output_len,
            max_batch_size=args.max_batch_size,
        )

        logging.info("Export is successful.")
    except Exception as error:
        logging.error("Error message: " + str(error))


if __name__ == '__main__':
    nemo_export(sys.argv[1:])
