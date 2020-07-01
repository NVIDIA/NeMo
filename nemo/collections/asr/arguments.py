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

from argparse import ArgumentParser


def add_asr_args(parent_parser: ArgumentParser) -> ArgumentParser:
    """Extends existing argparse with default ASR collection args.

    Args:
        parent_parser (ArgumentParser): Custom CLI parser that will be extended.

    Returns:
        ArgumentParser: Parser extended by NeMo ASR Collection arguments.
    """
    parser = ArgumentParser(parents=[parent_parser], add_help=False,)
    parser.add_argument("--asr_model", type=str, required=True, default="bad_quartznet15x5.yaml", help="")
    parser.add_argument("--train_dataset", type=str, required=True, default=None, help="training dataset path")
    parser.add_argument("--eval_dataset", type=str, required=True, help="evaluation dataset path")
    return parser
