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


def add_scheduler_args(parent_parser: ArgumentParser) -> ArgumentParser:
    """Extends existing argparse with default LR scheduler args.

    Args:
        parent_parser (ArgumentParser): Custom CLI parser that will be extended.

    Returns:
        ArgumentParser: Parser extended by LR Scheduler arguments.
    """
    parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
    parser.add_argument("--warmup_steps", type=int, required=False, default=None, help="Number of warmup steps")
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        required=False,
        default=None,
        help="Number of warmup steps as a percentage of total training steps",
    )
    parser.add_argument("--hold_steps", type=int, required=False, default=None, help="Number of hold LR steps")
    parser.add_argument(
        "--hold_ratio",
        type=float,
        required=False,
        default=None,
        help="Number of hold LR steps as a percentage of total training steps",
    )
    parser.add_argument("--min_lr", type=float, required=False, default=0.0, help="Minimum learning rate")
    parser.add_argument(
        "--last_epoch", type=int, required=False, default=-1, help="Last epoch id. -1 indicates training from scratch"
    )
    return parser
