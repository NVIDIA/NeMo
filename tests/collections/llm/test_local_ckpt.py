# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
Test NemMo 2.0 with local checkpointing from NVRx.
"""

import argparse
import logging
import socket
from dataclasses import dataclass
from pathlib import Path

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.model.llama import Llama3Config, LlamaModel
from nemo.collections.llm.recipes.log.default import get_global_step_from_global_checkpoint_path
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.lightning.pytorch.local_ckpt import update_trainer_local_checkpoint_io
from nemo.utils.import_utils import safe_import

res_module, HAVE_RES = safe_import("nvidia_resiliency_ext.ptl_resiliency")


@dataclass
class Llama3Config145M(Llama3Config):
    rotary_base: int = 500_000
    seq_length: int = 8192
    num_layers: int = 2
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16


def get_trainer(args, callbacks, async_save: bool = True) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=None,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=False,
        ckpt_async_save=async_save,
        ckpt_parallel_load=False,
        ddp=DistributedDataParallelConfig(),
    )
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        strategy=strategy,
    )
    return trainer


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Llama3 Pretraining on a local node")
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="How many nodes to use",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=2,
        help="Specify the number of GPUs per node",
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=200,
        help="Number of steps to run the training for",
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=150,
        help="Checkpoint saving interval in steps",
    )
    parser.add_argument(
        '--local-checkpoint-interval',
        type=int,
        default=60,
        help="Local checkpoint saving interval in steps",
    )
    parser.add_argument(
        '--val-check-interval',
        type=int,
        default=150,
        help="Validation check interval in steps",
    )
    parser.add_argument(
        '--limit_val_batches',
        type=int,
        default=10,
        help="How many batches to use for validation",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Output dir.",
        required=False,
        default="/tmp/nemo_llama3_local_ckpt",
    )
    parser.add_argument(
        "--async-save",
        action="store_true",
        help="Async ckpt save",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()

    assert HAVE_RES, "nvidia_resiliency_ext is required for local checkpointing"

    mbs = 1
    gbs = mbs * args.devices * args.num_nodes

    data = MockDataModule(
        seq_length=8192,
        global_batch_size=gbs,
        micro_batch_size=mbs,
    )

    model = LlamaModel(config=Llama3Config145M())

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        monitor="val_loss",
        save_top_k=1,
        every_n_train_steps=args.checkpoint_interval,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
        filename='{model_name}--{val_loss:.2f}-{step}-{consumed_samples}',
    )
    local_checkpoint_callback = res_module.local_checkpoint_callback.LocalCheckpointCallback(every_n_train_steps=args.local_checkpoint_interval)
    callbacks = [checkpoint_callback, local_checkpoint_callback]

    trainer = get_trainer(args, callbacks=callbacks, async_save=args.async_save)

    update_trainer_local_checkpoint_io(
        trainer,
        args.log_dir,
        get_global_step_from_global_checkpoint_path,
    )

    nemo_logger = nl.NeMoLogger(
        log_dir=args.log_dir,
        use_datetime_version=False,
        update_logger_directory=True,
        wandb=None,
        ckpt=checkpoint_callback,
    )

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-2,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        clip_grad=1.0,
        log_num_zeros_in_grad=False,
        timers=None,
        bf16=True,
        use_distributed_optimizer=False,
    )
    optim = MegatronOptimizerModule(config=opt_config)

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        resume=nl.AutoResume(
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
        ),
        optim=optim,
        tokenizer="data",
    )

    # Verify local checkpoints were created
    local_ckpt_dir = Path(args.log_dir) / "local_ckpt" / socket.gethostname()
    assert local_ckpt_dir.exists() and any(local_ckpt_dir.iterdir()), f"Expected local checkpoints in {local_ckpt_dir}, but directory is empty or doesn't exist"


if __name__ == "__main__":
    main()
