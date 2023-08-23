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

from nemo.export import TensorRTLLM
from nemo.utils import logging

try:
    from contextlib import nullcontext
except ImportError:
    # handle python < 3.7
    from contextlib import suppress as nullcontext

supported_model_types = ["gpt", "llama"]

def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Export models to TRT LLM",
    )
    parser.add_argument("--nemo_checkpoint", required=True, type=str, help="Source .nemo file")
    parser.add_argument(
        "--model_type",
        type=str, default="gpt",
        choices=["gpt", "llama"],
        help="Type of the model. gpt or llama are only supported."
    )
    parser.add_argument("--model_type", required=True, type=str, help="Type of the model. gpt or llama")
    parser.add_argument("--trt_llm_folder", default=None, type=str, help="Folder for the trt-llm conversion")
    parser.add_argument("--num_gpu", default=1, type=int, help="Number of GPUs that will deploy the model")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "fp8", "int8"],
        default="bfloat16",
        type=str,
        help="dtype of the model on TensorRT-LLM",
    )

    args = parser.parse_args(argv)
    return args


def nemo_convert(argv):
    args = get_args(argv)
    loglevel = logging.INFO
    logging.setLevel(loglevel)
    logging.info("Logging level set to {}".format(loglevel))
    logging.info(args)

    if args.dtype != "bfloat16":
        logging.error("Only bfloat16 is currently supported for the optimized deployment with TensorRT-LLM.")
        return

    trt_llm_path = args.trt_llm_folder

    if trt_llm_path is None:
        trt_llm_path = "/tmp/trt_llm_model_dir/"
        logging.info(
            "/tmp/trt_llm_model_dir/ path will be used as the TensorRT LLM folder. "
            "Please set this parameter if you'd like to use a path that has already"
            "included the TensorRT LLM model files."
        )
    if os.path.isfile(trt_llm_path):
        logging.error(
            "TensorRT LLM folder is pointing to a file. "
            "Please set this to a folder location."
        )
    Path(trt_llm_path).mkdir(parents=True, exist_ok=True)

    trt_llm_exporter = TensorRTLLM(model_dir=trt_llm_path)
    trt_llm_exporter.export(nemo_checkpoint_path=args.nemo_checkpoint, model_type=args.model_type, n_gpus=args.num_gpu)


if __name__ == '__main__':
    nemo_convert(sys.argv[1:])

