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
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.api import _setup, train
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, WarmupHoldPolicyScheduler

#                                 --ckpt-dir=/lustre/fsw/coreai_dlalgo_genai/ataghibakhsh/checkpoints/hyena_exp/small_ckpt \

"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 /opt/NeMo/tests/collections/llm/gpt/model/test_hyena.py \
                                --num-nodes=1 \
                                --devices=8 \
                                --max-steps=500000 \
                                --val-check-interval=200 \
                                --experiment-dir=/lustre/fsw/coreai_dlalgo_genai/ataghibakhsh/checkpoints/hyena_exp2 \
                                --data-path=/lustre/fsw/coreai_dlalgo_genai/ataghibakhsh/datasets/hyena_data/hg38/pretraining_data_hg38/data_hg38_all_text_CharLevelTokenizer_document \
                                --seq-length=8192 \
                                --tensor-parallel-size=1 \
                                --pipeline-model-parallel-size=1 \
                                --context-parallel-size=1 \
                                --global-batch-size=16 \
                                --micro-batch-size=2 \
                                --model-size=7b
"""

def get_args():
    parser = argparse.ArgumentParser(description='Train a Hyena model using NeMo 2.0')
    parser.add_argument('--num-nodes', type=int, default=1, help="Number of nodes to use for training, defaults to 1")
    parser.add_argument('--devices', type=int, help="Number of devices to use for training")
    parser.add_argument('--seq-length', type=int, default=8192, help="Training sequence length")
    parser.add_argument('--data-path', type=str, help="Data path")
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help="Tensor Parallel Size")
    parser.add_argument('--pipeline-model-parallel-size', type=int, default=1, help="Pipeline Parallel Size")
    parser.add_argument('--context-parallel-size', type=int, default=1, help="Context Parallel Size")
    parser.add_argument(
        "--sequence-parallel", action="store_true", help="Set to enable sequence parallel"
    )
    parser.add_argument('--micro-batch-size', type=int, default=1, help="Pipeline Parallel Size")
    parser.add_argument('--global-batch-size', type=int, default=8, help="Pipeline Parallel Size")
    parser.add_argument('--max-steps', type=int, help="Number of steps to train for")
    parser.add_argument('--val-check-interval', type=int, help="Number of steps between val check")
    parser.add_argument(
        '--model-size', type=str, default="7b", help="Model size, choose between 7b, 40b, or test (4 layers, less than 1b)"
    )
    parser.add_argument(
        '--experiment-dir', type=str, default=None, help="directory to write results to"
    )
    parser.add_argument(
        '--ckpt-dir', type=str, default=None, help="directory to write checkpoints to"
    )
    parser.add_argument('--tokenizer-path', type=str, default=None, help="Path to tokenizer model")

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    tokenizer = get_nmt_tokenizer(
        "byte-level",
    )

    data = PreTrainingDataModule(
        paths=args.data_path,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        seed=1234,
        num_workers=2,
        tokenizer=tokenizer,
    )

    if args.model_size == "7b":
        hyena_config = llm.Hyena7bConfig()
    elif args.model_size == "40b":
        hyena_config = llm.Hyena40bConfig()
    elif args.model_size == "test":
        hyena_config = llm.HyenaTestConfig()
    else:
        raise ValueError(f"Invalid model size: {args.model_size}")

    hyena_config.seq_length = args.seq_length
    model = llm.GPTModel(hyena_config, tokenizer=data.tokenizer)
    
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=args.val_check_interval,
        dirpath=args.experiment_dir,
        save_top_k=5,
        save_optim_on_train_end=True
    )
    
    loggers = []
    wandb_logger = WandbLogger(
        name=(f"hyena-size-{args.model_size}-TP{args.tensor_parallel_size}-"
        f"PP{args.pipeline_model_parallel_size}-CP{args.context_parallel_size}"
        f"-GBS{args.global_batch_size}-MBS{args.micro_batch_size}"),
        project="hyena_ux_test",
        save_dir=args.experiment_dir,
    )
    # wandb_logger = TensorBoardLogger(
    #     save_dir='dummy',  ## NOTE: this gets overwritten by default
    # )
    loggers.append(wandb_logger)

    nemo_logger = NeMoLogger(
        log_dir=args.experiment_dir,
        wandb=wandb_logger
    )

    trainer = nl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=args.tensor_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            context_parallel_size=args.context_parallel_size,
            pipeline_dtype=torch.bfloat16,
            sequence_parallel=args.sequence_parallel,
            ckpt_load_optimizer=False,
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            save_ckpt_format='zarr',
        ),
        logger=loggers,
        callbacks = [checkpoint_callback],
        log_every_n_steps=1,
        limit_val_batches=100,
        num_sanity_val_steps=0,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
        ),
        val_check_interval=args.val_check_interval,
    )

    # Logger setup
    nemo_logger.setup(
        trainer,
        resume_if_exists=True,
    )

    # Auto resume setup
    from nemo.lightning.pytorch.strategies.utils import RestoreConfig

    resume = nl.AutoResume(
        resume_if_exists=False,
        resume_ignore_no_checkpoint=True,
        resume_past_end=True,
        resume_from_directory=args.ckpt_dir,
        restore_config=(
            RestoreConfig(
                path=args.ckpt_dir,
                load_model_state = True,
                load_optim_state = False,
            ) if args.ckpt_dir else None
        ),
    )
    resume.setup(trainer, model)

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=0.0003,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        use_distributed_optimizer=True,
        bf16=True,
    )
    sched = CosineAnnealingScheduler(
        max_steps=trainer.max_steps,
        warmup_steps=2500,
        min_lr=0.00003,
    )

    opt = MegatronOptimizerModule(config=opt_config, no_weight_decay_cond=hyena_config.hyena_no_weight_decay_cond_fn, lr_scheduler=sched)
    # opt = MegatronOptimizerModule(opt_config, sched)
    opt.connect(model)

    # Start training
    trainer.fit(model, data)
