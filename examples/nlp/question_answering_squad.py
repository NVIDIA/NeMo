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

import random
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl

from nemo.collections.nlp.models.qa_model import QAModel
from nemo.core.optim.lr_scheduler import CosineAnnealing, SquareRootAnnealing, WarmupAnnealing
from nemo.utils.arguments import add_nlp_args, add_optimizer_args, add_scheduler_args


def add_args(parser):
    parser.add_argument("--max_epochs", default=10, type=int, help="Number of training epochs.")
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
        choices=["SquareRootAnnealing", "CosineAnnealing", "WarmupAnnealing"],
        help="Scheduler.",
    )
    parser.add_argument("--val_check_interval", default=1.0, type=int, help="validation after this many steps.")
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be "
        "generated. This is needed because the start "
        "and end predictions are not conditioned "
        "on one another.",
    )
    parser.add_argument(
        "--output_prediction_file",
        type=str,
        required=False,
        default="predictions.json",
        help="File to write predictions to. Only in evaluation or test mode.",
    )
    parser.add_argument(
        "--output_nbest_file",
        type=str,
        required=False,
        default="nbest.json",
        help="File to write nbest predictions to. Only in evaluation or test mode.",
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the examples contain some that do not have an answer.",
    )
    parser.add_argument(
        '--null_score_diff_threshold',
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )
    parser.add_argument(
        "--n_best_size", default=20, type=int, help="The total number of n-best predictions to generate at testing.",
    )
    parser.add_argument(
        "--mode",
        default="train_eval",
        choices=["train", "train_eval", "eval", "test"],
        help="Mode of model usage. When using test mode the script is running inference on the data, i.e. no ground-truth labels are required in the dataset.",
    )
    parser.add_argument(
        "--no_data_cache", action='store_true', help="When specified do not load and store cache preprocessed data.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. "
        "Questions longer than this will be truncated to "
        "this length.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after "
        "WordPiece tokenization. Sequences longer than this "
        "will be truncated, and sequences shorter than this "
        " will be padded.",
    )
    parser.add_argument(
        "--train_file", type=str, help="The training data file. Should be *.json",
    )
    parser.add_argument(
        "--eval_file", type=str, help="The evaluation data file. Should be *.json",
    )
    parser.add_argument(
        "--test_file", type=str, help="The test data file. Should be *.json. Does not need to contain ground truth",
    )
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training/evaluation.")
    parser.add_argument(
        "--use_cache", action='store_true', help="When specified do not load and store cache preprocessed data.",
    )
    args = parser.parse_args()
    return args


def main():
    parser = ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    parser = add_optimizer_args(parser)
    parser = add_scheduler_args(parser)
    parser = add_nlp_args(parser)
    args = add_args(parser)

    model = QAModel(
        pretrained_model_name=args.pretrained_model_name, config_file=args.config_file, num_classes=2, num_layers=1,
    )
    model.setup_training_data(
        train_data_layer_params={
            'data_file': args.train_file,
            'doc_stride': args.doc_stride,
            'max_query_length': args.max_query_length,
            'max_seq_length': args.max_seq_length,
            'version_2_with_negative': args.version_2_with_negative,
            'use_cache': args.use_cache,
            'batch_size': args.batch_size,
        },
    )
    model.setup_validation_data(
        val_data_layer_params={
            'data_file': args.eval_file,
            'doc_stride': args.doc_stride,
            'max_query_length': args.max_query_length,
            'max_seq_length': args.max_seq_length,
            'version_2_with_negative': args.version_2_with_negative,
            'use_cache': args.use_cache,
            'batch_size': args.batch_size,
        },
    )
    scheduler_args = {
        'monitor': 'val_loss',  # pytorch lightning requires this value
        'max_steps': args.max_steps,
    }
    if args.max_epochs:
        iters_per_batch = args.max_epochs / float(args.gpus * args.num_nodes * args.accumulate_grad_batches)
        scheduler_args['iters_per_batch'] = iters_per_batch
    else:
        scheduler_args['iters_per_batch'] = None

    scheduler_args["name"] = args.scheduler  # name of the scheduler
    scheduler_args["args"] = {
        "name": "auto",  # name of the scheduler config
        "params": {
            'warmup_ratio': args.warmup_ratio,
            'warmup_steps': args.warmup_steps,
            'last_epoch': args.last_epoch,
        },
    }

    model.setup_optimization(
        optim_params={
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
    trainer.fit(model)


if __name__ == '__main__':
    main()
