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

"""
Test fault tolerance with LLaMA3 recipe and a smaller model.
"""

import argparse
import os

import nemo_run as run
import torch

from nemo.collections import llm
from nemo.collections.llm.recipes.callbacks.default import straggler_det_callback
from nemo.lightning.run.plugins import FaultTolerancePlugin
from nemo.utils.exp_manager import TimingCallback
from tests.collections.llm.common import (
    create_verify_precision,
    small_llama_cfg,
    train_data,
    verify_ckpt_dir,
)


def get_args():
    parser = argparse.ArgumentParser(prog="", description="")
    parser.add_argument('--devices', type=int, required=True, help="Number of devices to use for training")
    parser.add_argument('--max-steps', type=int, required=True, help="Number of steps to train for")
    parser.add_argument(
        '--simulated-fault',
        action="store_true",
        default=False,
        help="Add a simulated fault to stop the training run (for testing nvidia-resiliency-ext's fault tolerance plugin)",
    )
    parser.add_argument(
        '--experiment-dir', type=str, required=True, help="directory to write results and checkpoints to"
    )
    parser.add_argument(
        '--data-path', type=str, default=None, help="Path to data file. If not specified, uses mock data."
    )
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        default=None,
        help="Path to a sentencepiece tokenizer model file. If not specified, uses mock data.",
    )
    parser.add_argument('--index-mapping-dir', type=str, help="directory to write index mappings to")
    
    return parser.parse_args()


def main():
    args = get_args()

    exp_name = "L2_llama3_small_pretrain_fault_tolerance_test"
    pretrain_recipe = llm.llama3_8b.pretrain_recipe(
        dir=args.experiment_dir, name=exp_name, num_gpus_per_node=args.devices
    )

    pretrain_recipe.model = run.Config(llm.LlamaModel, small_llama_cfg(1024))

    if args.data_path and args.tokenizer_path:
        pretrain_recipe.data = train_data(
            data_path=args.data_path,
            tokenizer_path=args.tokenizer_path,
            index_mapping_dir=args.index_mapping_dir,
            seq_length=1024,
        )

    # Recipe Overrides
    pretrain_recipe.trainer.max_steps = args.max_steps
    pretrain_recipe.trainer.log_every_n_steps = 1
    pretrain_recipe.log.ckpt.every_n_train_steps = None
    pretrain_recipe.log.ckpt.train_time_interval = None
    pretrain_recipe.trainer.val_check_interval = 2
    pretrain_recipe.trainer.limit_val_batches = 2

    
    executor: run.SlurmExecutor = run.LocalExecutor(ntasks_per_node=args.devices, launcher="ft")
    run_plugins: list[run.Plugin] = [FaultTolerancePlugin(), ConfigValidationPlugin(validate_preemption=False)]
    pretrain_recipe.trainer.callbacks = [run.Config(TimingCallback), straggler_det_callback()]

    run.run(pretrain_recipe, plugins=run_plugins, executor=executor)


if __name__ == '__main__':
    main()
