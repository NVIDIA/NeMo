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
import torch.nn as nn
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoProcessor

from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning.pytorch.optim import WarmupHoldPolicyScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback

from nemo.collections.diffusion.models.flux_controlnet.model import MegatronFluxControlNetModel, FluxControlNetConfig
from nemo.collections.diffusion.utils.flux_pipeline_utils import configs
from nemo.collections.diffusion.utils.mcore_parallel_utils import Utils
from megatron.core.distributed import DistributedDataParallelConfig
from nemo.collections.diffusion.data.diffusion_energon_datamodule import DiffusionDataModule
from nemo.collections.diffusion.data.diffusion_taskencoder import RawImageDiffusionTaskEncoder


def main(args):

    if args.use_synthetic_data:
        from nemo.collections.diffusion.data.diffusion_mock_datamodule import MockDataModule
        data = MockDataModule(
            image_h=1024,
            image_w=1024,
            micro_batch_size=args.mbs,
            global_batch_size=args.gbs,
            image_precached=args.image_precached,
            text_precached=args.text_precached,
        )
    else:
        data= DiffusionDataModule(
            args.dataset_dir,
            seq_length=4096,
            micro_batch_size=args.mbs,
            global_batch_size=args.gbs,
            num_workers=23,
            task_encoder=RawImageDiffusionTaskEncoder(),
        )

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1.0e-04,
        adam_beta1=0.9,
        adam_beta2=0.999,
        use_distributed_optimizer=True,
        bf16=True,
    )

    model_params = configs['dev']
    model_params.t5_params['version'] = '/ckpts/text_encoder_2'
    model_params.clip_params['version'] = '/ckpts/text_encoder'
    model_params.vae_params.ckpt = '/ckpts/ae.safetensors'

    if args.image_precached:
        model_params.vae_params = None
    if args.text_precached:
        model_params.t5_params = None
        model_params.clip_params = None

    flux_controlnet_config = FluxControlNetConfig(guidance_embed=True,num_joint_layers=args.num_joint_layers,num_single_layers=args.num_single_layers)

    model = MegatronFluxControlNetModel(model_params, flux_controlnet_config)

    ddp = DistributedDataParallelConfig(
        use_custom_fsdp=True,
        data_parallel_sharding_strategy='MODEL_AND_OPTIMIZER_STATES',
        overlap_param_gather=True,
        overlap_grad_reduce=True,
    )

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        ddp=ddp
    )


    # Checkpoint callback setup
    checkpoint_callback = nl.ModelCheckpoint(
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=2,
        every_n_train_steps=1000,
        dirpath=args.log_dir,
    )

    # Trainer setup
    trainer = nl.Trainer(
        num_nodes=args.num_nodes,
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        callbacks=[checkpoint_callback, TimingCallback()],
        val_check_interval=0,
        limit_val_batches=0,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    # Logger setup
    nemo_logger = nl.NeMoLogger(
        explicit_log_dir=args.log_dir,
        name=args.name,
        wandb=WandbLogger(project=args.wandb_project, name=args.name) if args.wandb_project is not None else None,
    )

    # Auto resume setup
    resume = nl.AutoResume(
        resume_if_exists=False,
        resume_ignore_no_checkpoint=True,
        resume_from_directory=args.log_dir,
        restore_config=nl.RestoreConfig(path=args.restore_path) if args.restore_path is not None else None,
    )


    sched = WarmupHoldPolicyScheduler(
        max_steps=trainer.max_steps,
        warmup_steps=1000,
        hold_steps=1000000000000,
    )
    opt = MegatronOptimizerModule(opt_config, sched)



    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        resume=resume,
        optim=opt
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--restore_path", type=str, required=False, default=None, help="Path to restore model from checkpoint"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=False,
        default="./nemo_experiments",
        help="Directory for logging and checkpoints",
    )
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--max_steps", type=int, required=False, default=5190)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--name", type=str, required=False, default="neva_pretrain")
    parser.add_argument("--wandb_project", type=str, required=False, default=None)
    parser.add_argument("--mbs", type=int, required=False, default=1)
    parser.add_argument("--gbs", type=int, required=False, default=1)
    parser.add_argument("--image_precached", action='store_true', default=False)
    parser.add_argument("--text_precached", action='store_true', default=False)
    parser.add_argument("--num_joint_layers", type=int, required=False, default=1)
    parser.add_argument("--num_single_layers", type=int, required=False, default=1)
    parser.add_argument("--use_synthetic_data", action='store_true', default=False)
    parser.add_argument("--dataset_dir", type=str, required=False, default=None)

    args = parser.parse_args()
    main(args)
