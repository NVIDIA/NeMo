
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


from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.utils.exp_manager import TimingCallback
from pytorch_lightning.loggers import WandbLogger
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from datetime import timedelta


def get_args():
    parser = argparse.ArgumentParser(description='BERT2 Pretraining in NeMo 2.0')
    parser.add_argument('--experiment_dir', type=str, help="directory to write results and checkpoints to")
    parser.add_argument('--devices', type=int, default=1, help="number of devices")
    parser.add_argument('--num_nodes', type=int, default=1, help="number of nodes")
    parser.add_argument('--max_steps', type=int, default=500000, help="max_steps")
    parser.add_argument('--mbs', type=int, default=4, help="micro batch size")
    parser.add_argument('--gbs', type=int, default=32, help="global batch size")
    parser.add_argument('--seq_length', type=int, default=512, help="Sequence length")
    parser.add_argument('--tp_size', type=int, default=1, help="tensor parallel size")
    parser.add_argument('--pp_size', type=int, default=1, help="pipeline parallel size")
    parser.add_argument('--dist_opt', type=int, default=0, help="whether to use dist_opt (1, 0)")
    parser.add_argument('--full_run', type=int, default=0, help="whether to use debug features")
    parser.add_argument('--resume', type=int, default=0, help="whether to resume if exists")
    parser.add_argument('--type', type=str, default='huggingface', help="huggingface|megatron")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--warmup', type=int, default=2000, help="warmup steps")
    parser.add_argument('--full', action='store_true', help="whether to use debug features")
    parser.add_argument('--lr', type=int, default=1, help="learning rate")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Logger
    ckpt = nl.ModelCheckpoint(
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=3,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
        train_time_interval=timedelta(minutes=30),
        filename='epoch={epoch}-step={step}'
    )
    if args.name is not None:
        wandb = WandbLogger(
            project=f"nemo2-esm2-tp",
            name=args.name,
        )
    else:
        wandb = None
    logger = nl.NeMoLogger(
        log_dir=args.experiment_dir,
        use_datetime_version=False,  # must be false if using auto resume
        ckpt=ckpt,
        wandb=wandb,
    )


    # Trainer
    callbacks = [TimingCallback()]
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_parallel_size=args.pp_size,
        context_parallel_size=1,
        sequence_parallel=False,
        gradient_as_bucket_view=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
    )
    trainer = nl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        log_every_n_steps=10,
        # limit_val_batches=10,
        # val_check_interval=100,
        limit_val_batches=0.0,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )

    adam = nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            optimizer="adam",
            lr=1e-4 * args.lr,
            adam_beta2=0.98,
            use_distributed_optimizer=True,
            clip_grad=1.0,
            bf16=True,
        ),
        lr_scheduler=CosineAnnealingScheduler(
            warmup_steps=args.warmup,
            constant_steps=0,
            min_lr=1e-5 * args.lr,
        )
    )


    # Model
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

    tokenizer = get_nmt_tokenizer("megatron", "facebook/esm2_t33_650M_UR50D", )

    # config = llm.ESM3BConfig()
    # config = llm.ESM650MConfig()
    config = llm.ESM15BConfig()
    config.seq_length = args.seq_length
    model = llm.BertModel(config, tokenizer=tokenizer)

    # Datamodule
    data = llm.BERTMockDataModule(
        seq_length=args.seq_length,
        micro_batch_size=args.mbs,
        global_batch_size=args.gbs,
        tokenizer=tokenizer,
        num_workers=0,
        num_train_samples=1_000_000_00,
    )

    # Resume
    resume = nl.AutoResume(
        # restore_config=nl.RestoreConfig(path=restore_path),
        resume_if_exists=args.resume,
        resume_ignore_no_checkpoint=True,
    )

    llm.pretrain(
        model=model,
        data=data,
        trainer=trainer,
        log=logger,
        optim=adam,
        resume=resume
    )
