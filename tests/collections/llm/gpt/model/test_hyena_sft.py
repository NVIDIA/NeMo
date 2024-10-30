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

## NOTE: This script is present for github-actions testing only.
## There are no guarantees that this script is up-to-date with latest NeMo.

import argparse

import torch
from megatron.core.optimizer import OptimizerConfig
from nemo.lightning.resume import AutoResume
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.api import _setup
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from nemo.collections.llm.gpt.data.mock import MockDataModule

"""
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 /opt/NeMo/tests/collections/llm/gpt/model/test_hyena_sft.py \
                                --devices=2 \
                                --max-steps=40 \
                                --experiment-dir=<path-to-experiment-dir> \
                                --seq-length=8192 \
                                --tensor-parallel-size=2 \
                                --pipeline-model-parallel-size=1 \
                                --global-batch-size=2 \
                                --micro-batch-size=1 \
                                --model-size=test \
                                --mode-path=<path-to-model>

"""

def get_args():
    parser = argparse.ArgumentParser(description='Train a Mamba model using NeMo 2.0')
    parser.add_argument('--devices', type=int, help="Number of devices to use for training")
    parser.add_argument('--seq-length', type=int, default=4096, help="Training sequence length")
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help="Tensor Parallel Size")
    parser.add_argument('--pipeline-model-parallel-size', type=int, default=1, help="Pipeline Parallel Size")
    parser.add_argument('--micro-batch-size', type=int, default=1, help="Pipeline Parallel Size")
    parser.add_argument('--global-batch-size', type=int, default=8, help="Pipeline Parallel Size")
    parser.add_argument('--max-steps', type=int, help="Number of steps to train for")
    parser.add_argument('--model-size', type=str, default="7b", help="Model size, choose between 7b or test (4 layers, less than 1b)")
    parser.add_argument(
        '--experiment-dir', type=str, default=None, help="directory to write results and checkpoints to"
    )
    parser.add_argument('--tokenizer-path', type=str, default=None, help="Path to tokenizer model")
    parser.add_argument('--model-path', type=str, default=None, help="Path to model to convert")

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()


    # Checkpoint callback setup
    checkpoint_callback = nl.ModelCheckpoint(
        every_n_train_steps=10,
        dirpath=args.experiment_dir,
    )

    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=nl.MegatronStrategy(
            ckpt_load_optimizer=False,
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            tensor_model_parallel_size=args.tensor_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            pipeline_dtype = torch.bfloat16,
        ),
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
        ),
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        limit_val_batches=5,
        val_check_interval=10,
        num_sanity_val_steps=0,
    )

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-5,
        min_lr=1e-5,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        bf16=True,
    )

    optim = MegatronOptimizerModule(config=opt_config)
    model_config = llm.HyenaTestConfig()
    model_config.seq_length = args.seq_length

    tokenizer = get_nmt_tokenizer(
        "byte-level",
    )
    model = llm.GPTModel(model_config, optim=optim, tokenizer=tokenizer)

    ckpt_path = model.import_ckpt(
        path="pytorch://" + args.model_path,
        model_config=model_config,
    )

    nemo_logger = NeMoLogger(
        log_dir=args.experiment_dir,
    )

    data = llm.SquadDataModule(
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        tokenizer=model.tokenizer,
        num_workers=0,
        pad_to_max_length=True,
    )

    # data = MockDataModule(
    #     seq_length=args.seq_length,
    #     tokenizer=tokenizer,
    #     micro_batch_size=args.micro_batch_size,
    #     global_batch_size=args.global_batch_size,
    #     num_train_samples=10_000,
    #     num_val_samples=10,
    #     num_test_samples=10,
    #     num_workers=0,
    #     pin_memory=False,
    # )

    ckpt_path = model.import_ckpt(
        path="pytorch://" + args.model_path,
        model_config=model_config,
    )

    app_state = _setup(
        model=model,
        data=data,
        resume=None,
        trainer=trainer,
        log=nemo_logger,
        optim=optim,
        tokenizer=tokenizer,
        model_transform=None,
    )

    trainer.fit(model, data, ckpt_path=ckpt_path)
