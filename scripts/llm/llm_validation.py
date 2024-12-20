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
from nemo.collections.llm.gpt.data.mock import MockDataModule
import torch
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import WandbLogger
from nemo.collections.nlp.models.language_modeling.megatron.gpt_layer_modelopt_spec import (
        get_gpt_layer_modelopt_spec,
    )
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

def main(args):
    # pylint: disable=C0115,C0116

    # Global and micro batch sizes
    gbs = args.gbs
    mbs = args.mbs
    max_steps = args.max_steps
    
    model_id = 'meta-llama/Llama-3.2-1B'
 
    data_module = MockDataModule(seq_length=256, global_batch_size=gbs, micro_batch_size=mbs)
    tokenizer = AutoTokenizer(model_id)

    llm_config =  llm.Llama32Config1B()
    llm_config.transformer_layer_spec=get_gpt_layer_modelopt_spec()
    
    model = llm.LlamaModel(llm_config, tokenizer=tokenizer)

    # Training strategy setup
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        encoder_pipeline_model_parallel_size=args.encoder_pp_size,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=False,
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
        max_steps=max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        callbacks=[checkpoint_callback, TimingCallback()],
        val_check_interval=500,
        limit_val_batches=gbs,
        log_every_n_steps=1,
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
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_from_directory=args.log_dir,
        restore_config=nl.RestoreConfig(path=args.restore_path) if args.restore_path is not None else None,
    )

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.95,
        use_distributed_optimizer=True,
        bf16=True,
    )
    sched = CosineAnnealingScheduler(
        max_steps=trainer.max_steps,
        warmup_steps=150,
        constant_steps=0,
        min_lr=2.0e-07,
    )
    opt = MegatronOptimizerModule(opt_config, sched)

    llm.validate(
        model=model,
        data=data_module,
        trainer=trainer,
        log=nemo_logger,
        optim=opt,
        resume=resume,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llava Next Finetuning Script")
    
    parser.add_argument(
        "--log_dir", type=str, required=False, default="/results", help="Directory for logging and checkpoints"
    )
    parser.add_argument(
        "--restore_path", type=str, required=False, default=None, help="Path to restore model from checkpoint"
    )
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--max_steps", type=int, required=False, default=5190)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--name", type=str, required=False, default="llm_validation_test")
    parser.add_argument("--wandb_project", type=str, required=False, default=None)
    parser.add_argument("--encoder_pp_size", type=int, required=False, default=0)
    parser.add_argument("--gbs", type=int, required=False, default=1, help="Global batch size")
    parser.add_argument("--mbs", type=int, required=False, default=1, help="Micro batch size")
    parser.add_argument("--lr", type=float, required=False, default=2.0e-05, help="Learning rate")

    args = parser.parse_args()
    main(args)
