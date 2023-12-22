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
import logging
from service.rest_model_api import app
import uvicorn

try:
    from contextlib import nullcontext
except ImportError:
    # handle python < 3.7
    from contextlib import suppress as nullcontext


LOGGER = logging.getLogger("NeMo")


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Deploy nemo models to Triton",
    )

    parser.add_argument(
        "-nc",
        "--nemo_checkpoint",
        type=str,
        help="Source .nemo file")

    parser.add_argument(
        "-pnc",
        "--ptuning_nemo_checkpoint",
        type=str,
        help="Source .nemo file for prompt embeddings table")

    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        required=False,
        choices=["gptnext", "gpt", "llama", "falcon", "starcoder"],
        help="Type of the model. gptnext, gpt, llama, falcon, and starcoder are only supported."
             " gptnext and gpt are the same and keeping it for backward compatibility"
    )

    parser.add_argument(
        "-tmn",
        "--triton_model_name",
        required=True,
        type=str,
        help="Name for the service"
    )

    parser.add_argument(
        "-tmv",
        "--triton_model_version",
        default=1,
        type=int,
        help="Version for the service"
    )

    parser.add_argument(
        "-tp",
        "--triton_port",
        default=8000,
        type=int,
        help="Port for the Triton server to listen for requests"
    )

    parser.add_argument(
        "-tha",
        "--triton_http_address",
        default="0.0.0.0",
        type=str,
        help="HTTP address for the Triton server"
    )

    parser.add_argument(
        "-tmr",
        "--triton_model_repository",
        default=None,
        type=str,
        help="Folder for the trt-llm conversion"
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
        choices=["bfloat16", "float16", "fp8", "int8"],
        default="bfloat16",
        type=str,
        help="dtype of the model on TensorRT-LLM",
    )

    parser.add_argument(
        "-mil",
        "--max_input_len",
        default=256,
        type=int,
        help="Max input length of the model"
    )

    parser.add_argument(
        "-mol",
        "--max_output_len",
        default=256,
        type=int,
        help="Max output length of the model"
    )

    parser.add_argument(
        "-mbs",
        "--max_batch_size",
        default=8,
        type=int,
        help="Max batch size of the model"
    )

    parser.add_argument(
        "-mpet",
        "--max_prompt_embedding_table_size",
        default=None,
        type=int,
        help="Max prompt embedding table size"
    )

    parser.add_argument(
        "-dcf",
        "--disable_context_fmha",
        action="store_true",
        help="Disable fused Context MultiHeadedAttention (required for V100 support)."
    )

    parser.add_argument(
        "-srs",
        "--start_rest_service",
        default="False",
        type=str,
        help="Starts the REST service for OpenAI API support"
    )

    parser.add_argument(
        "-sha",
        "--service_http_address",
        default="0.0.0.0",
        type=str,
        help="HTTP address for the REST Service"
    )

    parser.add_argument(
        "-sp",
        "--service_port",
        default=8000,
        type=int,
        help="Port for the Triton server to listen for requests"
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


def nemo_deploy(argv):
    args = get_args(argv)

    if args.debug_mode == "True":
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    LOGGER.setLevel(loglevel)
    LOGGER.info("Logging level set to {}".format(loglevel))
    LOGGER.info(args)

    if args.triton_model_repository is None:
        trt_llm_path = "/tmp/trt_llm_model_dir/"
        LOGGER.info(
            "/tmp/trt_llm_model_dir/ path will be used as the TensorRT LLM folder. "
            "Please set this parameter if you'd like to use a path that has already "
            "included the TensorRT LLM model files."
        )
        Path(trt_llm_path).mkdir(parents=True, exist_ok=True)
    else:
        trt_llm_path = args.triton_model_repository

    if args.nemo_checkpoint is None and args.triton_model_repository is None:
        LOGGER.error(
            "The provided model repository is not a valid TensorRT-LLM model "
            "directory. Please provide a --nemo_checkpoint."
        )
        return

    if args.nemo_checkpoint is None and not os.path.isdir(args.triton_model_repository):
        LOGGER.error(
            "The provided model repository is not a valid TensorRT-LLM model "
            "directory. Please provide a --nemo_checkpoint."
        )
        return

    if args.nemo_checkpoint is not None and args.model_type is None:
        LOGGER.error(
            "Model type is required to be defined if a nemo checkpoint is provided."
        )
        return

    if args.start_rest_service == "True":
        if args.service_port == args.triton_port:
            LOGGER.error(
                "REST service port and Triton server port cannot use the same port."
            )
            return

    trt_llm_exporter = TensorRTLLM(model_dir=trt_llm_path)

    if args.nemo_checkpoint is not None:
        try:
            LOGGER.info("Export operation will be started to export the nemo checkpoint to TensorRT-LLM.")
            trt_llm_exporter.export(
                nemo_checkpoint_path=args.nemo_checkpoint,
                model_type=args.model_type,
                n_gpus=args.num_gpus,
                max_input_token=args.max_input_len,
                max_output_token=args.max_output_len,
                enable_context_fmha=not args.disable_context_fmha,
                max_batch_size=args.max_batch_size,
                max_prompt_embedding_table_size=args.max_prompt_embedding_table_size,
                dtype=args.dtype
            )
        except Exception as error:
            LOGGER.error("An error has occurred during the model export. Error message: " + str(error))
            return

    try:
        nm = DeployPyTriton(
            model=trt_llm_exporter,
            triton_model_name=args.triton_model_name,
            triton_model_version=args.triton_model_version,
            max_batch_size=args.max_batch_size,
            port=args.triton_port,
            http_address=args.triton_http_address,
        )

        LOGGER.info("Triton deploy function will be called.")
        nm.deploy()
    except Exception as error:
        LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))
        return

    try:
        LOGGER.info("Model serving on Triton is will be started.")
        if args.start_rest_service == "True":
            nm.run()
        else:
            nm.serve()
    except Exception as error:
        LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))
        return

    if args.start_rest_service == "True":
        try:
            LOGGER.info("REST service is will be started.")
            uvicorn.run(
                'service.rest_model_api:app',
                host=args.service_http_address,
                port=args.service_port,
                reload=True
            )
        except Exception as error:
            LOGGER.error("Error message has occurred during REST service start. Error message: " + str(error))

    LOGGER.info("Model serving will be stopped.")
    nm.stop()


if __name__ == '__main__':
    nemo_deploy(sys.argv[1:])
