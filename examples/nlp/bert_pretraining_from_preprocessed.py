# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import sys
from argparse import ArgumentParser

import pytorch_lightning as pl

from nemo.collections.nlp.models.lm_model import BERTLMModel
from nemo.core.optim.lr_scheduler import CosineAnnealing, SquareRootAnnealing
from nemo.utils.arguments import add_nlp_args, add_optimizer_args, add_scheduler_args


def add_args(parser):
    parser.add_argument(
        "--gpus",
        default=1,
        help="Select GPU devices. Could be either int, string or list. See https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#select-gpu-devices",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size per worker for each model pass.")
    parser.add_argument(
        "--accumulate_grad_batches", default=1, type=int, help="Accumulates grads every k batches.",
    )
    parser.add_argument(
        "--amp_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Automatic Mixed Precision optimization level.",
    )
    parser.add_argument("--gradient_clip_val", type=float, default=0, help="gradient clipping")
    parser.add_argument(
        "--precision", default=32, type=int, choices=[16, 32], help="precision.",
    )
    parser.add_argument(
        "--scheduler",
        default='SquareRootAnnealing',
        type=str,
        choices=["SquareRootAnnealing", "CosineAnnealing"],
        help="Scheduler.",
    )
    parser.add_argument(
        "--max_predictions_per_seq",
        default=20,
        type=int,
        help="Maximum number of masked tokens to predict. Need to match the number of masked tokens in the input data sets.",
    )
    parser.add_argument(
        "--max_steps", default=100, type=int, help="Number of training steps.",
    )
    parser.add_argument("--pretrained_model_name", default='bert-base-uncased', type=str, help="pretrained model name")
    parser.add_argument(
        "--progress_bar_refresh_rate", default=1, type=int, help="progress_bar_refresh_rate",
    )
    parser.add_argument("--num_nodes", default=1, type=int, help="Number Nodes")
    args = parser.parse_args()
    return args


def main():
    parser = ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    parser = add_optimizer_args(parser)
    parser = add_scheduler_args(parser)
    parser = add_nlp_args(parser)
    args = add_args(parser)
    bert_model = BERTLMModel(pretrained_model_name=args.pretrained_model_name, config_file=args.config_file)
    bert_model.setup_training_data(
        train_data_layer_config={
            'train_data': args.data_dir,
            'max_predictions_per_seq': args.max_predictions_per_seq,
            'batch_size': args.batch_size,
        },
    )

    # Setup optimizer and scheduler
    scheduler_args = {
        'monitor': 'val_loss',  # pytorch lightning requires this value
        'max_steps': args.max_steps,
    }

    scheduler_args["name"] = args.scheduler  # name of the scheduler
    scheduler_args["args"] = {
        "name": "auto",  # name of the scheduler config
        "params": {
            'warmup_ratio': args.warmup_ratio,
            'warmup_steps': args.warmup_steps,
            'min_lr': args.min_lr,
            'last_epoch': args.last_epoch,
        },
    }

    bert_model.setup_optimization(
        optim_config={
            'name': args.optimizer,  # name of the optimizer
            'lr': args.lr,
            'args': {
                "name": "auto",  # name of the optimizer config
                "params": {},  # Put args.opt_args here explicitly
            },
            'sched': scheduler_args,
        }
    )
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        amp_level=args.amp_level,
        precision=args.precision,
        gpus=args.gpus,
        max_steps=args.max_steps,
        distributed_backend='ddp',
        replace_sampler_ddp=False,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
    )
    trainer.fit(bert_model)


if __name__ == '__main__':
    main()
