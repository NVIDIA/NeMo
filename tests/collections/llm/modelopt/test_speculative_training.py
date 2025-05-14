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

from argparse import ArgumentParser

import torch
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.modelopt import SpeculativeTransform
from nemo.lightning import io
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.lightning.pytorch.callbacks import ModelCheckpoint


def get_args():
    """Parse the command line arguments."""
    parser = ArgumentParser(description="""Run Knowledge Distillation from a teacher model to a student.""")

    parser.add_argument("--name", type=str, required=True, help="""Experiment name""")
    parser.add_argument("--model_path", type=str, required=True, help="""Path to NeMo 2 checkpoint""")
    parser.add_argument("--sd_algorithm", type=str, default="eagle", help="""Speculative decoding algorithm to use""")
    parser.add_argument("--tp_size", type=int, default=1, help="""Tensor parallel size""")
    parser.add_argument("--pp_size", type=int, default=1, help="""Pipeline parallel size""")
    parser.add_argument("--devices", type=int, default=1, help="""Number of GPUs to use per node""")
    parser.add_argument("--num_nodes", type=int, default=1, help="""Number of nodes to use""")
    parser.add_argument("--log_dir", type=str, required=True, help="""Folder for logging and checkpoint saving""")
    parser.add_argument("--max_steps", type=int, required=True, help="""Number of global batches to process""")
    parser.add_argument("--gbs", type=int, required=True, help="""Global Batch Size""")
    parser.add_argument("--mbs", type=int, required=True, help="""Micro-batch Size""")
    parser.add_argument("--seq_length", type=int, required=True, help="""Number of tokens per input sample""")
    parser.add_argument("--lr", type=float, default=1e-4, help="""Base LR for Cosine-Annealing scheduler""")
    parser.add_argument("--val_check_interval", type=int, default=100, help="""Validate + checkpoint every _ steps""")
    parser.add_argument("--limit_val_batches", type=int, default=32, help="""Number of batches per validation stage""")
    parser.add_argument("--log_interval", type=int, default=10, help="""Write to log every _ steps""")
    parser.add_argument("--legacy_ckpt", action="store_true", help="""Load ckpt saved with TE < 1.14""")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    ## Initialize the strategy and trainer
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        sequence_parallel=(args.tp_size > 1),
        ckpt_load_strictness=StrictHandling.LOG_ALL if args.legacy_ckpt else None,
    )
    trainer = nl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        log_every_n_steps=args.log_interval,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        strategy=strategy,
        accelerator="gpu",
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=True,
        ),
    )

    # Set up dataset
    data = llm.MockDataModule(
        seq_length=args.seq_length,
        global_batch_size=args.gbs,
        micro_batch_size=args.mbs,
    )

    ## Set up optimizer
    optim = nl.MegatronOptimizerModule(
        OptimizerConfig(
            optimizer="adam",
            lr=args.lr,
            bf16=True,
            use_distributed_optimizer=True,
        )
    )

    # Set up checkpointing and logging
    logger = nl.NeMoLogger(
        name=args.name,
        log_dir=args.log_dir,
        ckpt=ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            every_n_train_steps=args.val_check_interval,
        ),
    )

    # Load the model
    model = io.load_context(ckpt_to_context_subdir(args.model_path), subpath="model")
    model.config.make_vocab_size_divisible_by = 1
    model.config.gradient_accumulation_fusion = False

    # Run
    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=logger,
        optim=optim,
        model_transform=SpeculativeTransform(algorithm=args.sd_algorithm),
    )
