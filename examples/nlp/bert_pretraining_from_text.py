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

import json
import os
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl

from nemo.collections.nlp.models.lm_model import BERTLMModel
from nemo.core.optim.lr_scheduler import CosineAnnealing, SquareRootAnnealing
from nemo.utils.arguments import add_nlp_args, add_optimizer_args, add_scheduler_args


def add_args(parser):
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--sample_size", default=1e7, type=int, help="Data sample size.")
    parser.add_argument("--max_epochs", default=10, type=int, help="Number of training epochs.")
    parser.add_argument(
        "--mask_probability",
        default=0.15,
        type=float,
        help="Probability of masking a token in the input text during data processing.",
    )
    parser.add_argument(
        "--short_seq_prob",
        default=0.1,
        type=float,
        help="Probability of having a sequence shorter than the maximum sequence length `max_seq_length` in data processing.",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        help="Select GPU devices. Could be either int, string or list. See https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#select-gpu-devices",
    )
    parser.add_argument("--num_nodes", default=1, type=int, help="Number Nodes")
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
        "--max_steps", default=None, type=int, help="Number of training steps.",
    )
    parser.add_argument(
        "--precision", default=32, type=int, choices=[16, 32], help="precision.",
    )
    parser.add_argument(
        "--scheduler",
        default='CosineAnnealing',
        type=str,
        choices=["SquareRootAnnealing", "CosineAnnealing"],
        help="Scheduler.",
    )
    parser.add_argument("--val_check_interval", default=1.0, type=int, help="validation after this many steps.")
    args = parser.parse_args()
    return args


def main():
    parser = ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    parser = add_optimizer_args(parser)
    parser = add_scheduler_args(parser)
    parser = add_nlp_args(parser)
    args = add_args(parser)

    train_data_file = os.path.join(args.data_dir, 'train.txt')
    valid_data_file = os.path.join(args.data_dir, 'valid.txt')
    vocab_size = None
    if args.config_file:
        vocab_size = json.load(open(args.config_file))['vocab_size']
    preprocessing_args = {
        'data_file': train_data_file,
        'tokenizer_name': args.tokenizer_name,
        'vocab_size': vocab_size,
        'sample_size': args.sample_size,
        'pretrained_model_name': args.pretrained_model_name,
        'tokenizer_model': args.tokenizer_model,
        'do_lower_case': args.do_lower_case,
    }
    bert_model = BERTLMModel(
        pretrained_model_name=args.pretrained_model_name,
        config_file=args.config_file,
        preprocessing_args=preprocessing_args,
    )
    bert_model.setup_training_data(
        train_data_layer_config={
            'dataset': train_data_file,
            'batch_size': args.batch_size,
            'max_seq_length': args.max_seq_length,
            'mask_probability': args.short_seq_prob,
            'short_seq_prob': args.short_seq_prob,
        },
    )
    bert_model.setup_validation_data(
        val_data_layer_config={
            'dataset': valid_data_file,
            'batch_size': args.batch_size,
            'max_seq_length': args.max_seq_length,
            'mask_probability': args.short_seq_prob,
            'short_seq_prob': args.short_seq_prob,
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

    if args.max_steps is None:
        if args.gpus == 0:
            # training on CPU
            iters_per_batch = args.max_epochs / float(args.num_nodes * args.accumulate_grad_batches)
        else:
            iters_per_batch = args.max_epochs / float(args.gpus * args.num_nodes * args.accumulate_grad_batches)
        scheduler_args['iters_per_batch'] = iters_per_batch
    else:
        scheduler_args['iters_per_batch'] = None

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
        val_check_interval=args.val_check_interval,
        amp_level=args.amp_level,
        precision=args.precision,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        distributed_backend='ddp',
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
    )
    trainer.fit(bert_model)


if __name__ == '__main__':
    main()
