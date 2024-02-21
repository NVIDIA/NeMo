#!/usr/bin/python

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
import json
import logging
import os
import sys
import traceback
from builtins import range
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from cloud_telemetry_service import postToNVDataFlow

from nemo.deploy import DeployPyTriton, NemoQuery
from nemo.export import TensorRTLLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import statistics

try:
    from contextlib import nullcontext
except ImportError:
    # handle python < 3.7
    from contextlib import suppress as nullcontext


LOGGER = logging.getLogger("NeMo")


def nemo_deploy(args):
    loglevel = logging.DEBUG
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
        return None

    if args.nemo_checkpoint is None and not os.path.isdir(args.triton_model_repository):
        LOGGER.error(
            "The provided model repository is not a valid TensorRT-LLM model "
            "directory. Please provide a --nemo_checkpoint."
        )
        return None

    if args.nemo_checkpoint is not None and args.model_type is None:
        LOGGER.error("Model type is required to be defined if a nemo checkpoint is provided.")
        return None

    ptuning_tables_files = []
    if not args.ptuning_nemo_checkpoint is None:
        if args.max_prompt_embedding_table_size is None:
            LOGGER.error("max_prompt_embedding_table_size parameter is needed for the prompt tuning table(s).")
            return None

        ptuning_nemo_checkpoint_path = Path(args.ptuning_nemo_checkpoint)
        if ptuning_nemo_checkpoint_path.exists():
            if ptuning_nemo_checkpoint_path.is_file():
                ptuning_tables_files.append(args.ptuning_nemo_checkpoint)
            elif ptuning_nemo_checkpoint_path.is_dir():
                ptuning_tables_files.append(args.ptuning_nemo_checkpoint)
            else:
                LOGGER.error("Could not read the prompt tuning tables from {0}".format(args.ptuning_nemo_checkpoint))
                return None
        else:
            LOGGER.error("File or directory {0} does not exist.".format(args.ptuning_nemo_checkpoint))
            return None

    trt_llm_exporter = TensorRTLLM(model_dir=trt_llm_path)

    if args.nemo_checkpoint is not None:
        try:
            LOGGER.info("Export operation will be started to export the nemo checkpoint to TensorRT-LLM.")
            trt_llm_exporter.export(
                nemo_checkpoint_path=args.nemo_checkpoint,
                model_type=args.model_type,
                n_gpus=args.num_gpus,
                tensor_parallel_size=args.tensor_parallelism_size,
                pipeline_parallel_size=args.pipeline_parallelism_size,
                max_input_token=args.max_input_len,
                max_output_token=args.max_output_len,
                max_batch_size=args.max_batch_size,
                max_prompt_embedding_table_size=args.max_prompt_embedding_table_size,
                dtype=args.dtype,
                enable_multi_block_mode=args.multi_block_mode,
            )
        except Exception as error:
            LOGGER.error("An error has occurred during the model export. Error message: " + str(error))
            return None

    try:
        for task, prompt_embeddings_checkpoint_path in enumerate(ptuning_tables_files):
            LOGGER.info(
                "Adding prompt embedding table: {0} with task id: {1}.".format(prompt_embeddings_checkpoint_path, task)
            )
            trt_llm_exporter.add_prompt_table(
                task_name=str(task), prompt_embeddings_checkpoint_path=prompt_embeddings_checkpoint_path,
            )
    except Exception as error:
        LOGGER.error("An error has occurred during adding the prompt embedding table(s). Error message: " + str(error))
        return None

    try:
        nm = DeployPyTriton(
            model=trt_llm_exporter,
            triton_model_name=args.triton_model_name,
            triton_model_version=args.triton_model_version,
            max_batch_size=args.max_batch_size,
            port=args.triton_port,
            address=args.triton_http_address,
        )

        LOGGER.info("Triton deploy function will be called.")
        nm.deploy()
    except Exception as error:
        LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))
        return None

    try:
        LOGGER.info("Model serving on Triton will be started.")
        nm.run()
    except Exception as error:
        LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))
        return None

    return nm


def get_inputs():
    # Parse the json file - use "rb" to make sure that JSON parser understands UTF-8 characters
    with open("/opt/NeMo/scripts/deploy/benchmark_data.json", "rb") as json_file:
        data = json.load(json_file)

    return {
        128: [data["test_input_128"]],
        256: [data["test_input_256"]],
        512: [data["test_input_512"]],
        2048: [data["test_input_2048"]],
    }


def perform_benchmark(args):
    nq = NemoQuery(url="localhost:8000", model_name=args.triton_model_name)

    input_info = get_inputs()

    for prompt_len, prompt in input_info.items():
        if prompt_len <= args.max_input_len:
            for out_len in args.output_lens:
                for batch_size in args.batch_size:

                    LOGGER.info(
                        "Starting to get measurements for "
                        "prompt len: {0}, output len {1}, and batch size "
                        "{2}".format(prompt_len, out_len, batch_size)
                    )

                    batch_size = int(batch_size)
                    inputs = prompt * batch_size

                    failed = False

                    # warm up
                    if args.warm_up:
                        output = nq.query_llm(prompts=inputs, max_output_token=out_len)
                        if output[0][0].startswith("An error occurred"):
                            failed = True

                    latencies = []
                    for i in range(args.num_runs):
                        start_time = datetime.now()
                        output = nq.query_llm(prompts=inputs, max_output_token=out_len)
                        stop_time = datetime.now()

                        if output[0][0].startswith("An error occurred"):
                            failed = True
                            LOGGER.warning("Got an error from the last query. Error message: {0}".format(output[0][0]))
                            break

                        latencies.append((stop_time - start_time).total_seconds() * 1000.0)

                    if not failed:
                        if args.num_runs > 1:
                            latency = statistics.mean(latencies)
                        else:
                            latency = latencies[0]

                        latency = round(latency, 3)
                        throughput = round(1000 / latency * batch_size, 3)

                        LOGGER.info(
                            "Latency: {0} ms, and throughput: {1} prompts/sec, for "
                            "prompt len: {2}, output len: {3}, and batch size: "
                            "{4}.".format(latency, throughput, prompt_len, out_len, batch_size)
                        )

                        if args.ci_upload_test_results_to_cloud:
                            postToNVDataFlow({"latency": latency})
                    else:
                        latency = None

                    if args.out_jsonl:
                        measurement = {
                            "failed": failed,
                            "batch_size": batch_size,
                            "input_len": prompt_len,
                            "output_len": out_len,
                            "latency": latency,
                            "all_latencies": latencies,
                            "nemo_checkpoint_path": args.nemo_checkpoint,
                            "model_type": args.model_type,
                            "n_gpus": args.num_gpus,
                            "device": torch.cuda.get_device_name(),
                            "device_properties": str(torch.cuda.get_device_properties(0)),
                            "full_args": vars(args),
                        }

                        def custom_serializer(obj):
                            try:
                                return json.JSONEncoder().default(obj)
                            except TypeError:
                                return f"Unserializable: {type(obj).__name__}"

                        args.out_jsonl.write(json.dumps(measurement, default=custom_serializer) + "\n")
                        args.out_jsonl.flush()


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Deploy nemo models to Triton and benchmark the models",
    )

    parser.add_argument("-nc", "--nemo_checkpoint", type=str, help="Source .nemo file")
    parser.add_argument(
        "-pnc", "--ptuning_nemo_checkpoint", type=str, help="Source .nemo file for prompt embeddings table"
    )
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        required=False,
        choices=["gptnext", "gpt", "llama", "falcon", "starcoder", "gemma"],
        help="Type of the model. gptnext, gpt, llama, falcon, and starcoder are only supported."
        " gptnext and gpt are the same and keeping it for backward compatibility",
    )
    parser.add_argument("-tmn", "--triton_model_name", required=True, type=str, help="Name for the service")
    parser.add_argument("-tmv", "--triton_model_version", default=1, type=int, help="Version for the service")
    parser.add_argument(
        "-tp", "--triton_port", default=8000, type=int, help="Port for the Triton server to listen for requests"
    )
    parser.add_argument(
        "-tha", "--triton_http_address", default="0.0.0.0", type=str, help="HTTP address for the Triton server"
    )
    parser.add_argument(
        "-tmr", "--triton_model_repository", default=None, type=str, help="Folder for the trt-llm conversion"
    )
    parser.add_argument("-ng", "--num_gpus", default=1, type=int, help="Number of GPUs for the deployment")
    parser.add_argument("-tps", "--tensor_parallelism_size", type=int, help="Tensor parallelism size")
    parser.add_argument("-pps", "--pipeline_parallelism_size", type=int, help="Pipeline parallelism size")
    parser.add_argument(
        "-dt",
        "--dtype",
        choices=["bfloat16", "float16", "fp8", "int8"],
        default="bfloat16",
        type=str,
        help="dtype of the model on TensorRT-LLM",
    )
    parser.add_argument("-mil", "--max_input_len", default=512, type=int, help="Max input length of the model")
    parser.add_argument("-mol", "--max_output_len", default=512, type=int, help="Max output length of the model")
    parser.add_argument("-mbs", "--max_batch_size", default=8, type=int, help="Max batch size of the model")
    parser.add_argument(
        "-mpet", "--max_prompt_embedding_table_size", default=None, type=int, help="Max prompt embedding table size"
    )
    parser.add_argument(
        '-w', '--warm_up', action="store_true", required=False, default=False, help='Enable warm_up before benchmark'
    )
    parser.add_argument(
        '-bs', '--batch_size', nargs='+', default=["1", "2", "4", "8"], required=False, help='Specify batch size'
    )
    parser.add_argument(
        "-mbm",
        '--multi_block_mode',
        default=False,
        action='store_true',
        help='Split long kv sequence into multiple blocks (applied to generation MHA kernels). \
                        It is beneifical when batchxnum_heads cannot fully utilize GPU.',
    )
    parser.add_argument(
        '-ol',
        '--output_lens',
        nargs='+',
        default=[20, 100, 200, 300],
        type=int,
        required=False,
        help='Lengths of outputs',
    )
    parser.add_argument('-nr', '--num_runs', type=int, default=8, required=False, help='Specify input length')
    parser.add_argument(
        '-rt',
        '--run_trt_llm',
        choices=[0, 1],
        type=int,
        default=0,
        required=False,
        help='Run TRT-LLM without PyTriton',
    )
    parser.add_argument('--out_jsonl', type=argparse.FileType('w'), required=False)
    parser.add_argument(
        '-ci',
        '--ci_upload_test_results_to_cloud',
        action="store_true",
        required=False,
        default=False,
        help='Post telemetry service data for ci runs',
    )

    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    if args.max_output_len < args.output_lens[-1]:
        raise Exception(
            "max_output_len is set to {0} and cannot be lower "
            "than the max value in output_lens which is "
            "{1}".format(args.max_output_len, args.output_lens[-1])
        )

    nm = nemo_deploy(args)

    if nm is None:
        LOGGER.info("There is a problem and model serving couldn't be started. Please check the log messages.")
    else:
        try:
            perform_benchmark(args)
        except Exception as e:
            LOGGER.error("An error has occurred while sending queries.")
            LOGGER.error(repr(e))
            LOGGER.error(traceback.format_exc())
        try:
            LOGGER.info("Model serving will be stopped.")
            nm.stop()
        except Exception as e:
            LOGGER.error("Model could not be stopped properly.")
            LOGGER.error(repr(e))
            LOGGER.error(traceback.format_exc())
