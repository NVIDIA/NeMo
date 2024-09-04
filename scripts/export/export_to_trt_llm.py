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

import logging
import sys

from nemo.export.tensorrt_llm import TensorRTLLM
from nemo.export.tensorrt_llm_parsing_utils import add_export_kwargs, add_multi_block_mode_flag, create_parser

LOGGER = logging.getLogger("NeMo")


def get_args(argv):
    parser = create_parser(
        "Exports nemo models stored in nemo checkpoints to TensorRT-LLM", export_parser=True, deploy=False
    )
    parser = add_multi_block_mode_flag(parser)
    parser.add_argument(
        "-mr", "--model_repository", required=True, default=None, type=str, help="Folder for the trt-llm model files"
    )
    parser.add_argument("-dm", "--debug_mode", default=False, action='store_true', help="Enable debug mode")
    args = parser.parse_args(argv)
    return add_export_kwargs(args, deploy=False)


def nemo_export_trt_llm(argv):
    args = get_args(argv)

    loglevel = logging.DEBUG if args.debug_mode else logging.INFO
    LOGGER.setLevel(loglevel)
    LOGGER.info("Logging level set to {}".format(loglevel))
    LOGGER.info(args)

    if args.dtype != "bfloat16":
        LOGGER.error(
            "Only bf16 is currently supported for the optimized deployment with TensorRT-LLM. "
            "Support for the other precisions will be added in the coming releases."
        )
        return

    try:
        trt_llm_exporter = TensorRTLLM(
            model_dir=args.model_repository, load_model=False, multi_block_mode=args.multi_block_mode
        )
        LOGGER.info("Export to TensorRT-LLM function is called.")
        trt_llm_exporter.export(**args.export_kwargs)
        LOGGER.info("Export is successful.")
    except Exception as error:
        LOGGER.error("Error message: " + str(error))
        raise error


if __name__ == '__main__':
    nemo_export_trt_llm(sys.argv[1:])
