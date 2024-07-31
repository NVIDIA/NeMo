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

import argparse
import torch
import shutil
import os

from pytorch_lightning.loggers import TensorBoardLogger

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.api import train
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.collections.llm.tools.auto_configurator import AutoConfigurator
from nemo.collections.llm.tools.auto_configurator import get_results
from nemo.utils.exp_manager import TimingCallback


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_number", type=int, help="Number of config to run")
    parser.add_argument("--logs_dir", type=str, help="Path where to save training logs")
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--get_results", action="store_true")

    return parser.parse_args()

def train_config(args):
    # GPT-3 126M
    # This example will generate 3 configs.
    # It is expected that this script will be run 3 times with changing --run_number flag for each run from 0 to 2.
    # After all configurations are trained, please trigger the script using --get_results flag.
    runner = AutoConfigurator(
        model_type="gpt3",
        model_version=3,
        model_size=126,
        model_measure="M",
        num_nodes=1,
        gpus_per_node=1,
        gpu_memory_gb=40,
        global_batch_size=16,
        seq_length=512,
        tensor_parallel_sizes=[1],
        pipeline_parallel_sizes=[1],
        micro_batch_sizes=[1, 2, 4],
        max_training_days=1,
        max_steps_per_run=25,
        num_tokens_in_b=10,
        vocab_size=51200,
        data_paths=args.data_path,
    )

    # Get generated configs
    configs = runner.generate_configs()

    tokenizer = get_nmt_tokenizer(
        "megatron",
        "GPT2BPETokenizer",
    )

    # Define candidate to run
    runs = list(configs.keys())
    run_name = runs[args.run_number]
    config = configs[run_name]

    # Define data
    config["data"].pop('split')
    data = PreTrainingDataModule(**config["data"], split="900,50,50", tokenizer=tokenizer)

    # Define model
    config["model"].tensor_model_parallel_size = config["auto_config"].get("tensor_model_parallel_size")
    config["model"].pipeline_model_parallel_size = config["auto_config"].get("pipeline_model_parallel_size")
    config["model"].context_parallel_size = config["auto_config"].get("context_parallel_size")
    config["model"].expert_model_parallel_size = config["auto_config"].get("expert_model_parallel_size")
    model = llm.GPTModel(config["model"], tokenizer=data.tokenizer)

    # Define optimizer
    opt = MegatronOptimizerModule(config=config["optim"])

    # Define strategy
    strategy = nl.MegatronStrategy(
        pipeline_dtype=torch.bfloat16,
        tensor_model_parallel_size=config["auto_config"].get("tensor_model_parallel_size"),
        pipeline_model_parallel_size=config["auto_config"].get("pipeline_model_parallel_size"),
        virtual_pipeline_model_parallel_size=config["auto_config"].get("virtual_pipeline_model_parallel_size", None),
        context_parallel_size=config["auto_config"].get("context_parallel_size"),
        expert_model_parallel_size=config["auto_config"].get("expert_model_parallel_size"),
    )

    # Define TensorBoard logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"{args.logs_dir}/{run_name}",
    )

    # Define trainer
    trainer = nl.Trainer(
        **config["trainer"],
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        callbacks=[TimingCallback()],
        logger=tensorboard_logger,
    )

    # Define logger
    nemo_logger = NeMoLogger(
        dir=f"{args.logs_dir}/{run_name}",
    )

    # Train candidate
    train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer="data",
        optim=opt,
    )

def main():
    args = get_args()

    if not args.get_results:
        train_config(args)
    
    else:
        # Get Auto Configurator results
        candidates = [d for d in os.listdir(args.logs_dir) if os.path.isdir(os.path.join(args.logs_dir, d))]
        for subdir in candidates:
            default_dir = os.path.join(args.logs_dir, subdir, "default")
            if os.path.exists(default_dir) and os.path.isdir(default_dir):
                for item in os.listdir(default_dir):
                    s = os.path.join(default_dir, item)
                    d = os.path.join(args.logs_dir, subdir, item)
                    shutil.move(s, d)

                os.rmdir(default_dir)
        
        get_results(
            training_logs=args.logs_dir,
            path_to_save=args.logs_dir,
            model_name="gpt3",
            model_version=3,
            model_size=126,
            model_measure="M",
            num_nodes=1,
            gpus_per_node=1,
            global_batch_size=16,
            seq_length=512,
            max_training_days=1,
            num_tokens_in_b=10,
            vocab_size=51200,
        )
        
        print(f"The results were successfully saved to {args.logs_dir}.")

if __name__ == '__main__':
    main()