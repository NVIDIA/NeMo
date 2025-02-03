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
import time

from PIL import ImageFile
from lightning import Callback

from nemo.lightning.io import pl
from nemo.utils import logging, timers

ImageFile.LOAD_TRUNCATED_IMAGES = True
# BaseWebdataset.
import webdataset
import pdb, os
# from megatron.energon.wrappers.iter_map_dataset import

from nemo.collections.multimodal.data.clip.clip_dataset import build_imagenet_validation_dataloader_params
from nemo.collections.multimodal.data.energon.base import EnergonMultiModalDataModule


# pdb.set_trace = lambda: 1
"""
Example:
  python scripts/vlm/clip_finetune.py \
  --data_type=mock
"""

import argparse

import torch
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import WandbLogger

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.diffusion.data.diffusion_energon_datamodule import DiffusionDataModule
from nemo.collections.vlm import ImageDataConfig
from nemo.collections.vlm.clip.data.clip_data_module import RawImageDiffusionTaskEncoder, ClipTaskEncoder
from nemo.collections.vlm.clip.model import ClipConfigL14, ClipConfigB32, CLIPModel
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback, DataLoaderProfilingCallback


def return_false(a,b):
    return False

def main(args):
    # pylint: disable=C0115,C0116
    from importlib.metadata import version
    assert version("megatron.energon") == '3.0.1.dev136+g920bb6b', ("Please use the dev energon."
                                                                    "pip install 'megatron-energon @ git+https://github.com/NVIDIA/Megatron-Energon.git@920bb6b430f115fc0d9e75900bd39bfb21335ed9'")
    # Global and micro batch sizes
    gbs = args.mbs * args.devices * args.num_nodes
    args.gbs = gbs
    mbs = args.mbs
    max_steps = args.max_steps

    decoder_seq_length = 80
    task_encoder = ClipTaskEncoder(max_length=decoder_seq_length)
    if True:
# DiffusionDataModule
        data = EnergonMultiModalDataModule(
            args.data_path,
            seq_length=decoder_seq_length,
            image_processor=None,
            micro_batch_size=args.mbs,
            global_batch_size=args.gbs,
            num_workers=args.num_workers,
            task_encoder=task_encoder,
            tokenizer = task_encoder.tokenizer,
        )
    elif args.data_type == "mock":
        data = vlm.ClipMockDataModule(
            seq_length=decoder_seq_length,
            global_batch_size=gbs,
            micro_batch_size=mbs,
            tokenizer=None,
            num_train_samples=10_000_000_000,
            image_processor=None,
            num_workers=8,
        )
    else:
        raise ValueError(f"Data type {args.data_type} not supported")

    config = ClipConfigB32()
    # imagenet_validation_dataloader = None
    # imagenet_validation_dataloader = build_imagenet_validation_dataloader_params( args.imagenet_val,
    #     config.vision_transformer_config.img_h,
    #                                                                              config.vision_transformer_config.img_w,
    #                                                                               mbs, gbs, num_workers=4,
    #                                                                               max_position_embedding=decoder_seq_length,
    #                                                                               tokenizer=task_encoder.tokenizer
    #                                                                              )


    model = CLIPModel(ClipConfigB32(), tokenizer=task_encoder.tokenizer,
                      imagenet_val=args.imagenet_val,
                      mbs=mbs,
                      gbs=gbs, max_workers=8)
    # , imagenet_val_dataloader=imagenet_validation_dataloader)
    # model.set_imageval_dataloader(imagenet_validation_dataloader)
    from megatron.core.distributed import DistributedDataParallelConfig

    # Training strategy setup
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        encoder_pipeline_model_parallel_size=args.encoder_pp_size,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=True,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=True,
            average_in_collective=True,
        ),
    )
    strategy=nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        ckpt_async_save=False,
        ddp=DistributedDataParallelConfig(
            overlap_grad_reduce=True,
            check_for_nan_in_grad=True,
            overlap_param_gather=True,
        )
    )


    # Checkpoint callback setup
    checkpoint_callback = nl.ModelCheckpoint(
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=2,
        save_on_train_epoch_end=True, # This prevents it from saving after validation
        every_n_train_steps=25000,
        dirpath=os.path.join(args.log_dir, args.name),
    )

    # Trainer setup
    trainer = nl.Trainer(
        num_nodes=args.num_nodes,
        devices=args.devices,
        max_steps=max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        callbacks=[checkpoint_callback, TimingCallback()],
        # val_check_interval=1000,
        check_val_every_n_epoch=5,
        limit_val_batches=1,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )

    # Logger setup
    nemo_logger = nl.NeMoLogger(
        log_dir=args.log_dir,
        name=args.name,
        wandb=WandbLogger(project=args.wandb_project, name=args.name) if args.wandb_project is not None else None,
    )

    # Auto resume setup
    resume = nl.AutoResume(
        resume_if_exists=False,
        resume_ignore_no_checkpoint=True,
        resume_from_directory=os.path.join(args.log_dir, args.name),
    )

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-3,
        adam_beta1=0.9,
        adam_beta2=0.98,
        weight_decay=0.2,
    )


    sched = CosineAnnealingScheduler(
        max_steps=trainer.max_steps,
        warmup_steps=2000,
        constant_steps=0,
        min_lr=1e-5,
    )
    opt = MegatronOptimizerModule(opt_config, sched,
                                  no_weight_decay_cond=return_false,)


    llm.pretrain(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        optim=opt,
        resume=resume,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEVA Model Training Script")

    # Argument parsing
    parser.add_argument("--data_type", type=str, required=False, default="mock", help="mock | llava")
    parser.add_argument("--data_path", type=str, required=False, default="/workspace/data/cc3m_training_single_sample", help="Path to the dataset JSON file")
    parser.add_argument("--image_folder", type=str, required=False, default=None, help="Path to the image folder")
    parser.add_argument(
        "--log_dir", type=str, required=False, default="/results", help="Directory for logging and checkpoints"
    )

    parser.add_argument(
        "--language_model_path", type=str, required=False, default=None, help="Path to the pretrained language model"
    )
    parser.add_argument(
        "--restore_path", type=str, required=False, default=None, help="Path to restore model from checkpoint"
    )
    parser.add_argument("--mbs", type=int, required=False, default=32, help="Micro batch size")
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--num_workers", type=int, required=False, default=8)

    parser.add_argument("--max_steps", type=int, required=False, default=375000)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--encoder_pp_size", type=int, required=False, default=0)
    parser.add_argument("--projector_type", type=str, required=False, default="mcore_mlp")
    parser.add_argument("--name", type=str, required=False, default="clip_pretrain")
    parser.add_argument("--peft", type=str, default='none', help="none | lora")
    parser.add_argument("--wandb_project", type=str, required=False, default=None)
    parser.add_argument("--gbs", type=int, required=False, default=64, help="Global batch size")
    parser.add_argument("--lr", type=float, required=False, default=2.0e-06, help="Learning rate")
    parser.add_argument("--imagenet_val", type=str, required=False, default="/workspace/data/imagenet_validation", help="imagenet path for val")

    args = parser.parse_args()
    main(args)

