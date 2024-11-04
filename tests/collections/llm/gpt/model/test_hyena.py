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
from pytorch_lightning.loggers import TensorBoardLogger

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.api import _setup, train
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.api import _setup


"""
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 /opt/NeMo/tests/collections/llm/gpt/model/test_hyena.py \
                                --devices=2 \
                                --max-steps=40 \
                                --experiment-dir=/home/ataghibakhsh/temp_ckpt \
                                --seq-length=8192 \
                                --tensor-parallel-size=2 \
                                --pipeline-model-parallel-size=1 \
                                --global-batch-size=2 \
                                --micro-batch-size=1 \
                                --model-size=test
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
    parser.add_argument(
        '--model-size', type=str, default="7b", help="Model size, choose between 7b or test (4 layers, less than 1b)"
    )
    parser.add_argument(
        '--experiment-dir', type=str, default=None, help="directory to write results and checkpoints to"
    )
    parser.add_argument('--tokenizer-path', type=str, default=None, help="Path to tokenizer model")
    # parser.add_argument('--data-path', type=str, default=None, help="Path to data file")

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    tokenizer = get_nmt_tokenizer(
        "byte-level",
    )

    data = MockDataModule(
        seq_length=args.seq_length,
        tokenizer=tokenizer,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        num_train_samples=10_000,
        num_val_samples=10,
        num_test_samples=10,
        num_workers=0,
        pin_memory=False,
    )

    if args.model_size == "7b":
        hyena_config = llm.Hyena7bConfig()
    elif args.model_size == "test":
        hyena_config = llm.HyenaTestConfig()
    else:
        raise ValueError(f"Invalid model size: {args.model_size}")

    hyena_config.seq_length = args.seq_length
    model = llm.GPTModel(hyena_config, tokenizer=data.tokenizer)
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        pipeline_dtype=torch.bfloat16,
        ckpt_load_optimizer=False,
        ckpt_save_optimizer=False,
        ckpt_async_save=False,
    )
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=10,
        dirpath=args.experiment_dir,
    )
    callbacks = [checkpoint_callback]

    loggers = []
    tensorboard_logger = TensorBoardLogger(
        save_dir='dummy',  ## NOTE: this gets overwritten by default
    )
    loggers.append(tensorboard_logger)

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=6e-4,
        min_lr=6e-5,
        clip_grad=1.0,
        use_distributed_optimizer=True,
        bf16=True,
    )
    opt = MegatronOptimizerModule(config=opt_config)

    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=1,
        limit_val_batches=2,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
        ),
    )

    nemo_logger = NeMoLogger(
        log_dir=args.experiment_dir,
    )

    app_state = _setup(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        optim=opt,
        resume=None,
        tokenizer='data',
        model_transform=None,
    )
    trainer.fit(model, data)
