# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0115,C0116,C0301

import os
from argparse import ArgumentParser

import torch
from cosmos1.models.autoregressive.nemo.cosmos_video2world import (
    CosmosConfigVideo2World5B,
    CosmosConfigVideo2World13B,
    CosmosVideo2WorldModel,
)
from cosmos1.models.autoregressive.nemo.post_training.video2world_dataset import CosmosVideo2WorldDataModule
from huggingface_hub import snapshot_download
from lightning.pytorch.loggers import WandbLogger
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning.pytorch.callbacks import ModelCheckpoint, PreemptionCallback
from nemo.lightning.pytorch.strategies.utils import RestoreConfig


def main(args):
    if "5B" in args.model_path or "5b" in args.model_path:
        model_config = CosmosConfigVideo2World5B()
    elif "13B" in args.model_path or "13b" in args.model_path:
        model_config = CosmosConfigVideo2World13B()
    else:
        raise NotImplementedError

    if args.model_path in [
        "nvidia/Cosmos-1.0-Autoregressive-5B-Video2World",
        "nvidia/Cosmos-1.0-Autoregressive-13B-Video2World",
    ]:
        args.model_path = os.path.join(snapshot_download(args.model_path, allow_patterns=["nemo/*"]), "nemo")

    model = CosmosVideo2WorldModel(model_config)

    data_module = CosmosVideo2WorldDataModule(
        data_path=args.data_path,
        model_config=model_config,
        seq_length=12864,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        tokenizer=None,
        num_workers=1,
    )

    # Finetune is the same as train (Except train giverm the option to set tokenizer to None)
    # So we use it since in this case we dont store a tokenizer with the model
    llm.api.train(
        model=model,
        data=data_module,
        trainer=nl.Trainer(
            devices=args.tensor_model_parallel_size,
            num_nodes=1,
            max_steps=args.max_steps,
            accelerator="gpu",
            strategy=nl.MegatronStrategy(
                tensor_model_parallel_size=args.tensor_model_parallel_size,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                sequence_parallel=False,
                pipeline_dtype=torch.bfloat16,
            ),
            plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
            num_sanity_val_steps=0,
            limit_val_batches=0,
            max_epochs=args.max_epochs,
            log_every_n_steps=1,
            callbacks=[
                ModelCheckpoint(
                    monitor="reduced_train_loss",
                    filename="{epoch}-{step}",
                    every_n_train_steps=args.save_every_n_steps,
                    save_top_k=2,
                ),
                PreemptionCallback(),
            ],
        ),
        log=nl.NeMoLogger(wandb=(WandbLogger() if "WANDB_API_KEY" in os.environ else None), log_dir=args.log_dir),
        optim=nl.MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=args.lr,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=False,
            )
        ),
        tokenizer=None,
        resume=nl.AutoResume(
            restore_config=RestoreConfig(path=args.model_path),
            resume_if_exists=True,
            resume_ignore_no_checkpoint=False,
            resume_past_end=True,
        ),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str, help="The path to the .pt files")
    parser.add_argument(
        "--model_path",
        default="nvidia/Cosmos-1.0-Autoregressive-5B-Video2World",
        type=str,
        help="The path to the nemo model",
    )

    parser.add_argument("--log_dir", default="./log_dir", type=str, help="The path to the logs")
    parser.add_argument("--tensor_model_parallel_size", default=4, type=int, help="Tensor model parallel size")
    parser.add_argument("--max_steps", default=10, type=int, help="The max number of steps to run finetuning")
    parser.add_argument("--save_every_n_steps", default=5, type=int, help="How often to save a checkpoint")
    parser.add_argument("--global_batch_size", default=1, type=int, help="The global batch size")
    parser.add_argument(
        "--micro_batch_size", default=1, type=int, help="The micro batch size if using pipeline parallel"
    )
    parser.add_argument("--lr", default=5e-5, type=float, help="The learning rate")
    parser.add_argument("--max_epochs", default=10, type=int, help="Max number of epochs")

    args = parser.parse_args()

    main(args)
